import logging
import re
from pathlib import Path
import os


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

def prepare_lama():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    if os.environ.get("LAMA_MODEL", None) == None:
        os.environ["LAMA_MODEL"] = "data/models/AnimeMangaInpainting/model.jit.pt"
    else:
        return

    os.makedirs("data/models/AnimeMangaInpainting", exist_ok=True)
    for hub_file in [
        "model.jit.pt",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/AnimeMangaInpainting" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="s9roll74/tracing_dreMaz_AnimeMangaInpainting", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/AnimeMangaInpainting"
        )
    
    if False:
        from bubble_tool.inpainting_lama_mpe import LamaFourier
        def load_lama_mpe(model_path, device, use_mpe: bool = True, large_arch: bool = False) -> LamaFourier:
            model = LamaFourier(build_discriminator=False, use_mpe=use_mpe, large_arch=large_arch)
            sd = torch.load(model_path, map_location = 'cpu')
            model.generator.load_state_dict(sd['gen_state_dict'])
            if use_mpe:
                model.mpe.load_state_dict(sd['str_state_dict'])
            model.eval().to(device)
            return model

        model = load_lama_mpe("data/models/AnimeMangaInpainting/lama_large_512px.ckpt",  device='cpu', use_mpe=False, large_arch=True)
        traced_model = torch.jit.trace(model.generator, ( torch.zeros([1, 3, 512, 512]), torch.zeros([1,1,512, 512])))
        traced_model.save("data/models/AnimeMangaInpainting/model.jit.pt")


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



#############################################################################

def detect_bubble(src_list, mask_path:Path, model_type, detection_th, classification):

    if model_type == 0:
        prepare_yolo()

    if len(YOLO_SEG_MODEL_LOCATION) > model_type:
        model = YOLO(YOLO_SEG_MODEL_LOCATION[model_type])
    else:
        raise ValueError(f"unknown {model_type=}")

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    for i, src in tqdm( enumerate(src_list), desc=f"detect_bubble", total=len(src_list)):

        result_path = mask_path / Path(Path(src).with_suffix(".png").name)

        org_size = Image.open(src).size

        with torch.no_grad():
            results = model.predict(src, save=False, verbose=False, conf=detection_th)

        masks = results[0].masks
        boxes = results[0].boxes

        result = [None,None,None]

        if masks is not None:
            for mask, box in zip(masks,boxes):
                cls = int(box.cls)
                if cls > 2:
                    cls = 2

                mask = scale_image_torch(mask.data, (org_size[1],org_size[0]))
                if result[cls] is not None:
                    result[cls] += mask.squeeze()
                else:
                    result[cls] = mask.squeeze()
            
            for i in range(len(result)):
                if result[i] is not None:
                    result[i] = result[i].cpu().numpy()
                    result[i] = result[i].astype('uint8') * 255
                else:
                    result[i] = np.zeros((org_size[1],org_size[0]), np.uint8)
            
            if classification:
                result_array = np.dstack(result)
            else:
                result_array = result[0] | result[1] | result[2]

            Image.fromarray(result_array).save(result_path)
        else:
            if classification:
                result_array = np.zeros((org_size[1],org_size[0],3), np.uint8)
            else:
                result_array = np.zeros((org_size[1],org_size[0]), np.uint8)

            Image.fromarray( result_array ).save(result_path)
            

        if False:
            bubble_array = np.array(Image.open(src))
            bubble_array[result==0] = 120

            Image.fromarray(bubble_array).save("bubble_only.png")
    
    model.to("cpu")

    torch.cuda.empty_cache()



def lama_inpaint(src_list, mask_list, output_path:Path):

    prepare_lama()

    simple_lama = SimpleLama()

    for i, (src, mask) in tqdm( enumerate(zip(src_list,mask_list)), desc=f"lama_inpaint", total=min(len(src_list),len(mask_list))):

        result_path = output_path / Path(Path(src).name)

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

def blend_image_C(org_array, mask_array, dst_array):

    org_array[mask_array==0] = (0)

    gray = cv2.cvtColor(org_array, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros((org_array.shape[0],org_array.shape[1]), np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255), -1)

    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    
    mask = mask / 255

    dst_array = org_array * mask + dst_array * (1 - mask)

    return dst_array


def blend_image(src_list, mask_list, base_list, output_path:Path, blend_method):

    for i, (src, mask, base) in tqdm( enumerate(zip(src_list,mask_list,base_list)), desc=f"blend_image", total=min(len(src_list),len(mask_list),len(base_list))):

        result_path = output_path / Path(Path(src).name)

        org_array = np.array(Image.open(src))
        mask_array = np.array(Image.open(mask))
        dst_array = np.array(Image.open(base))
        org_array = org_array[:,:,:3]
        dst_array = dst_array[:,:,:3]

        if blend_method != -1:
            if mask_array.ndim == 3:
                mask_array = mask_array[:, :, 0] | mask_array[:, :, 1] | mask_array[:, :, 2]

        # DEBUG
        #dst_array[:,:,:] = 125

        org_array = resize_img( org_array, (dst_array.shape[1], dst_array.shape[0]) )
        mask_array = resize_img( mask_array, (dst_array.shape[1], dst_array.shape[0]) )

        if blend_method == -1:

            dst_array = blend_image_A(org_array.copy(), mask_array[:, :, 0], dst_array)
            dst_array = blend_image_B(org_array.copy(), mask_array[:, :, 1], dst_array)
            dst_array = blend_image_B(org_array.copy(), mask_array[:, :, 2], dst_array)

        elif blend_method == 0:
            dst_array = blend_image_A(org_array, mask_array, dst_array)
        elif blend_method == 1:
            dst_array = blend_image_B(org_array, mask_array, dst_array)
        else:
            dst_array = blend_image_C(org_array, mask_array, dst_array)

        Image.fromarray(dst_array.astype(np.uint8)).save(result_path)


def remove_bubble(src_path:Path, mask_output_path:Path, clean_img_output_path:Path, model_type, detection_th):

    src_list = get_image_file_list(src_path)

    detect_bubble(src_list, mask_output_path, model_type, detection_th, False)

    mask_list = get_image_file_list(mask_output_path)

    lama_inpaint(src_list, mask_list, clean_img_output_path)



def copy_bubble(src_path:Path, base_path:Path, mask_input_path:Path, create_mask, with_bubble_img_output_path:Path, model_type, blend_method, detection_th):

    base_list = get_image_file_list(base_path)
    src_list = get_image_file_list(src_path)

    src_map = { s.stem : s for s in src_list }

    src_list = []

    for b in base_list:
        src_img_path = src_map.get(b.stem, None)

        if src_img_path:
            if src_img_path.is_file():
                src_list.append( src_img_path )


    if create_mask:
        detect_bubble(src_list, mask_input_path, model_type, detection_th, True)

    mask_list = [ (mask_input_path/Path(s.name)).with_suffix(".png") for s in base_list if (mask_input_path/Path(s.name)).with_suffix(".png").is_file() ]

    blend_image(src_list, mask_list, base_list, with_bubble_img_output_path, blend_method)



def split_panel(src_path:Path, split_output_path:Path, output_size):
    from transformers import AutoModel

    src_list = get_image_file_list(src_path)

    def read_image_as_np_array(image_path):
        with open(image_path, "rb") as file:
            image = Image.open(file).convert("L").convert("RGB")
            image = np.array(image)
        return image
    
    images = [read_image_as_np_array(image) for image in src_list]

    model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cuda()
    with torch.no_grad():
        results = model.predict_detections_and_associations(images)

    crop_info = []

    for i, (result, src_path) in enumerate(zip(results, src_list)) :
        with Image.open(src_path) as im:
            crop_info.append(f"{src_path.name},{im.size[0]},{im.size[1]}")
            for j, panel in enumerate(result["panels"]):
                im_crop = im.crop(panel)
                scale = output_size / (im_crop.size[0] + im_crop.size[1])
                im_crop = im_crop.resize( (int(im_crop.size[0] * scale), int(im_crop.size[1] * scale)), Image.LANCZOS )
                filename = str(i).zfill(8) + "_" + str(j).zfill(8) + ".png"
                im_crop.save( split_output_path / Path(filename) )
                crop_info.append(f"{filename},{panel[0]},{panel[1]},{panel[2]},{panel[3]}")
    
    crop_info_path = split_output_path / Path("split_info.txt")
    crop_info_path.write_text( "\n".join(crop_info), encoding="utf-8")

    model.to("cpu")

    torch.cuda.empty_cache()


def combine_panel(src_path:Path, fragment_path:Path, output_dir:Path):

    crop_info_path = fragment_path / Path("split_info.txt")
    if crop_info_path.is_file() == False:
        raise ValueError(f"{crop_info_path} not found")
    
    crop_list = crop_info_path.read_text()
    crop_list = crop_list.splitlines()

    def create_crop_info(crop_list):
        crop_info = {}
        cur_src_name = None
        cur_item = {}

        for c in crop_list:
            c = c.split(",")
            if len(c) == 3:
                if cur_item:
                    crop_info[cur_src_name] = cur_item
                cur_item = {}
                cur_src_name, x, y = c
                cur_item["org_size"] = (int(x), int(y))
                cur_item["frags"] = {}
            else:
                frag_name, x1, y1, x2, y2 = c
                x1 = int(float(x1))
                y1 = int(float(y1))
                x2 = int(float(x2))
                y2 = int(float(y2))
                cur_item["frags"][frag_name] = (x1,y1,x2,y2)

        if cur_item:
            crop_info[cur_src_name] = cur_item
        return crop_info
    

    crop_info = create_crop_info(crop_list)
    

    for i, src_name in enumerate(crop_info):
        src_img_path = src_path / Path(src_name)
        if src_img_path.is_file() == False:
            continue
        src_img = Image.open( src_img_path )
        org_size = crop_info[src_name]["org_size"]

        mod = False

        for frag_name in crop_info[src_name]["frags"]:
            frag_img_path = fragment_path / Path(frag_name)
            if frag_img_path.is_file() == False:
                frag_img_path = frag_img_path.with_suffix(".png")
                if frag_img_path.is_file() == False:
                    continue
            frag_img = Image.open(frag_img_path)
            x1,y1,x2,y2 = crop_info[src_name]["frags"][frag_name]

            scale_x = (x2-x1) / org_size[0]
            scale_y = (y2-y1) / org_size[1]
            frag_img = frag_img.resize( (int(src_img.size[0] * scale_x), int(src_img.size[1] * scale_y)), Image.LANCZOS )
            scale_x = src_img.size[0]/org_size[0]
            scale_y = src_img.size[1]/org_size[1]
            src_img.paste(frag_img, (int(x1 * scale_x),int(y1 * scale_y)) )
            mod = True

        if mod:
            src_img.save( output_dir/ Path(src_name) )

