import logging
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

#############################################################################
YOLO_SEG_MODEL_LOCATION = [                                         # model_type
    "data/models/yolov8m_seg-speech-bubble/model.pt",               # 0
    "data/models/adetailerForTextSpeech_v20/unwantedV10x.pt",       # 1 (Manual download is required)
]

#############################################################################

BUBBLE_BORDER_COLOR = (0,0,0)       # black
BUBBLE_BORDER_WIDTH = 0             # 0 pixel -> auto-calculation

#############################################################################

def get_image_file_list(img_dir_path:Path):
    img_list = [p for p in img_dir_path.glob("*") if re.search(r'.*\.(jpg|png|webp)', str(p))]
    return sorted(img_list)

def resize_img(img, size_xy):
    if img.shape[0] > size_xy[1]:
        return cv2.resize(img, size_xy, interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, size_xy, interpolation=cv2.INTER_CUBIC)

def prepare_yolo():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/yolov8m_seg-speech-bubble", exist_ok=True)
    for hub_file in [
        "model.pt",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/yolov8m_seg-speech-bubble" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="kitsumed/yolov8m_seg-speech-bubble", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/yolov8m_seg-speech-bubble"
        )

# https://github.com/ultralytics/ultralytics/issues/3560
def scale_image_torch(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size

    Args:
      masks (torch.Tensor): resized and padded masks/images, [c, h, w].
      im0_shape (tuple): the original image shape
      ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
      masks (torch.Tensor): The masks that are being returned.
    """
    import torch.nn.functional as F

    if len(masks.shape) < 3:
        raise ValueError(
            f'"len of masks shape" should be 3, but got {len(masks.shape)}'
        )
    im1_shape = masks.shape
    if im1_shape[1:] == im0_shape:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(
            im1_shape[1] / im0_shape[0], im1_shape[2] / im0_shape[1]
        )  # gain  = old / new
        pad = (im1_shape[2] - im0_shape[1] * gain) / 2, (
            im1_shape[1] - im0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[1] - pad[1]), int(im1_shape[2] - pad[0])

    masks = masks[:, top:bottom, left:right]
    if masks.shape[1:] != im0_shape:
        masks = F.interpolate(
            masks[None], im0_shape, mode="bilinear", align_corners=False
        )[0]

    return masks



def detect_bubble(src_list, mask_path:Path, model_type, detection_th):

    if model_type == 0:
        prepare_yolo()

    if len(YOLO_SEG_MODEL_LOCATION) > model_type:
        model = YOLO(YOLO_SEG_MODEL_LOCATION[model_type])
    else:
        raise ValueError(f"unknown {model_type=}")

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    for i, src in tqdm( enumerate(src_list), desc=f"detect_bubble", total=len(src_list)):

        result_path = mask_path / Path(str(i).zfill(8) + ".png")

        org_size = Image.open(src).size

        with torch.no_grad():
            results = model.predict(src, save=False, verbose=False, conf=detection_th)

        masks = results[0].masks
        boxes = results[0].boxes

        result = None

        if masks is not None:
            for mask, box in zip(masks,boxes):
                #logger.info(f"{box.conf=}")
                mask = scale_image_torch(mask.data, (org_size[1],org_size[0]))
                if result is not None:
                    result += mask.squeeze()
                else:
                    result = mask.squeeze()
            
            result = result.cpu().numpy()
            result = result.astype('uint8') * 255

            Image.fromarray(result).save(result_path)
        else:
            Image.fromarray( np.zeros((org_size[1],org_size[0]), np.uint8) ).save(result_path)

        if False:
            bubble_array = np.array(Image.open(src))
            bubble_array[result==0] = 120

            Image.fromarray(bubble_array).save("bubble_only.png")
    
    model.to("cpu")

    torch.cuda.empty_cache()



def lama_inpaint(src_list, mask_list, output_path:Path):

    simple_lama = SimpleLama()

    for i, (src, mask) in tqdm( enumerate(zip(src_list,mask_list)), desc=f"lama_inpaint", total=min(len(src_list),len(mask_list))):

        result_path = output_path / Path(str(i).zfill(8) + ".png")

        image = Image.open(src)

        image = image.convert('RGB')

        org_size = image.size

        mask = np.array(Image.open(mask).convert('L'))

        k = int(org_size[0] * 10 / 480)
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), 3)

        k = int(org_size[0] * 9 / 480) //2 * 2 + 1
        mask = cv2.GaussianBlur(mask, ksize=(k,k), sigmaX=0)

        mask[mask >= 125] = 255
        mask[mask < 125] = 0

        result = simple_lama(image, mask)
        result.save(result_path)
    
    simple_lama.model.to("cpu")

    torch.cuda.empty_cache()


def blend_image_A(org_array, mask_array, dst_array):

    kernel1_size = int(30 / 1000 * dst_array.shape[1])
    kernel1_size = max(kernel1_size , 3)

    kernel2_size = int(5 / 1000 * dst_array.shape[1])
    kernel2_size = max(kernel2_size , 3)

    gaussian_k_size = int(31 / 1000 * dst_array.shape[1]) // 2 * 2 + 1
    gaussian_k_size = max(gaussian_k_size , 3)

    bubble_border_width = int(3 / 1000 * dst_array.shape[1])
    bubble_border_width = max(bubble_border_width , 1)
    if BUBBLE_BORDER_WIDTH:
        bubble_border_width = BUBBLE_BORDER_WIDTH

    kernel = np.ones((kernel1_size,kernel1_size),np.uint8)
    kernel2 = np.ones((kernel2_size,kernel2_size),np.uint8)

    mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
    mask_array = cv2.erode(mask_array,kernel2,iterations = 1)

    mask_array = cv2.GaussianBlur(mask_array, ksize=(gaussian_k_size,gaussian_k_size), sigmaX=0)
    mask_array[mask_array >= 125] = 255
    mask_array[mask_array < 125] = 0

    #Image.fromarray(mask_array.astype(np.uint8)).save("new_mask.png")

    contours, _ = cv2.findContours(mask_array, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for j in range(len(contours)):
        if cv2.contourArea(contours[j]) > (mask_array.shape[0] * mask_array.shape[1]) * 0.005:
            org_array = cv2.drawContours(org_array, contours, j, BUBBLE_BORDER_COLOR, bubble_border_width)

    if mask_array.ndim == 2:
        mask_array = mask_array[:, :, np.newaxis]

    mask_array = mask_array / 255

    dst_array = org_array * mask_array + dst_array * (1 - mask_array)

    return dst_array

def blend_image_B(org_array, mask_array, dst_array):

    gaussian_k_size = int(31 / 1000 * dst_array.shape[1]) // 2 * 2 + 1

    mask_array = cv2.GaussianBlur(mask_array, ksize=(gaussian_k_size,gaussian_k_size), sigmaX=0)

    if mask_array.ndim == 2:
        mask_array = mask_array[:, :, np.newaxis]

    mask_array = mask_array / 255

    dst_array = org_array * mask_array + dst_array * (1 - mask_array)

    return dst_array

def blend_image(src_list, mask_list, base_list, output_path:Path, blend_method):

    for i, (src, mask, base) in tqdm( enumerate(zip(src_list,mask_list,base_list)), desc=f"blend_image", total=min(len(src_list),len(mask_list),len(base_list))):

        result_path = output_path / Path(str(i).zfill(8) + ".png")

        org_array = np.array(Image.open(src))
        mask_array = np.array(Image.open(mask))
        dst_array = np.array(Image.open(base))
        org_array = org_array[:,:,:3]
        dst_array = dst_array[:,:,:3]

        org_array = resize_img( org_array, (dst_array.shape[1], dst_array.shape[0]) )
        mask_array = resize_img( mask_array, (dst_array.shape[1], dst_array.shape[0]) )

        if blend_method == 0:
            dst_array = blend_image_A(org_array, mask_array, dst_array)
        else:
            dst_array = blend_image_B(org_array, mask_array, dst_array)

        Image.fromarray(dst_array.astype(np.uint8)).save(result_path)


def remove_bubble(src_path:Path, mask_output_path:Path, clean_img_output_path:Path, model_type, detection_th):

    src_list = get_image_file_list(src_path)

    detect_bubble(src_list, mask_output_path, model_type, detection_th)

    mask_list = get_image_file_list(mask_output_path)

    lama_inpaint(src_list, mask_list, clean_img_output_path)



def copy_bubble(src_path:Path, base_path:Path, mask_input_path:Path, create_mask, with_bubble_img_output_path:Path, model_type, blend_method, detection_th):

    src_list = get_image_file_list(src_path)

    if create_mask:
        detect_bubble(src_list, mask_input_path, model_type, detection_th)

    base_list = get_image_file_list(base_path)
    mask_list = get_image_file_list(mask_input_path)

    blend_image(src_list, mask_list, base_list, with_bubble_img_output_path, blend_method)

