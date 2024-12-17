import logging
import time
from pathlib import Path
import fire
from datetime import datetime

from bubble_tool.bubble_tool import remove_bubble,copy_bubble

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_proj_dirs(proj_dir_path:Path):
    (proj_dir_path / Path("src")).mkdir(parents=True, exist_ok=True)
    (proj_dir_path / Path("mask")).mkdir(parents=True, exist_ok=True)
    (proj_dir_path / Path("cleaned")).mkdir(parents=True, exist_ok=True)
    (proj_dir_path / Path("base")).mkdir(parents=True, exist_ok=True)
    (proj_dir_path / Path("base_with_bubble")).mkdir(parents=True, exist_ok=True)


class Command:
    def __init__(self):
        pass

    def remove(self, src_path, model_type:int=0, detection_th:float=0.1):
        # src image -> mask + clean image
        start_tim = time.time()

        src_path = Path(src_path)
        if not src_path.is_dir():
            raise ValueError( f"{src_path} not found" )        

        output_dir = Path("output") / Path("remove") / Path(get_time_str())

        mask_output_path = output_dir / Path("mask")
        mask_output_path.mkdir(parents=True)
        clean_img_output_path = output_dir / Path("cleaned_img")
        clean_img_output_path.mkdir(parents=True)

        remove_bubble(src_path, mask_output_path, clean_img_output_path, model_type, detection_th)

        logger.info(f"Output : {clean_img_output_path}")
        logger.info(f"Total Elapsed time : {time.time() - start_tim}")


    def copy(self, src_path, base_path, mask_path=None, model_type:int=0, blend_method:int=0, detection_th:float=0.1):
        # src image, base_image ,(bubble_mask) -> base_with_bubble image
        start_tim = time.time()

        src_path = Path(src_path)
        if not src_path.is_dir():
            raise ValueError( f"{src_path} not found" )
        
        base_path = Path(base_path)
        if not base_path.is_dir():
            raise ValueError( f"{base_path} not found" )

        output_dir = Path("output") / Path("copy") / Path(get_time_str())

        with_bubble_img_output_path =output_dir / Path("with_bubble_img")
        with_bubble_img_output_path.mkdir(parents=True)

        if mask_path:
            mask_path = Path(mask_path)
            if not mask_path.is_dir():
                raise ValueError( f"{mask_path} not found" )
            
            creat_mask = False

        else:
            mask_path = output_dir / Path("mask")
            mask_path.mkdir(parents=True)

            creat_mask = True

        copy_bubble(src_path, base_path, mask_path, creat_mask, with_bubble_img_output_path, model_type, blend_method, detection_th)

        logger.info(f"Output : {with_bubble_img_output_path}")
        logger.info(f"Total Elapsed time : {time.time() - start_tim}")


    def remove_proj(self, proj_dir_path, model_type:int=0, detection_th:float=0.1):
        # src image -> mask + clean image
        start_tim = time.time()

        proj_dir_path = Path(proj_dir_path)
        if not proj_dir_path.is_dir():
            raise ValueError( f"{proj_dir_path} not found" )
        
        src_path = proj_dir_path / Path("src")
        if not src_path.is_dir():
            raise ValueError( f"{src_path} not found" )
        
        create_proj_dirs(proj_dir_path)

        time_str = get_time_str()
        
        mask_output_path = proj_dir_path / Path("mask") / Path(time_str)
        mask_output_path.mkdir(parents=True)
        clean_img_output_path = proj_dir_path / Path("cleaned") / Path(time_str)
        clean_img_output_path.mkdir(parents=True)

        remove_bubble(src_path, mask_output_path, clean_img_output_path, model_type, detection_th)

        logger.info(f"Output : {clean_img_output_path}")
        logger.info(f"Total Elapsed time : {time.time() - start_tim}")


    def copy_proj(self, proj_dir_path, model_type:int=0, blend_method:int=0, detection_th:float=0.1):
        # src image, base_image ,(bubble_mask) -> base_with_bubble image
        start_tim = time.time()

        proj_dir_path = Path(proj_dir_path)
        if not proj_dir_path.is_dir():
            raise ValueError( f"{proj_dir_path} not found" )
        
        src_path = proj_dir_path / Path("src")
        if not src_path.is_dir():
            raise ValueError( f"{src_path} not found" )
        
        base_path = proj_dir_path / Path("base")
        if not base_path.is_dir():
            raise ValueError( f"{base_path} not found" )
        
        create_proj_dirs(proj_dir_path)

        time_str = get_time_str()
        
        mask_output_path = proj_dir_path / Path("mask") / Path(time_str)
        mask_output_path.mkdir(parents=True)
        with_bubble_img_output_path = proj_dir_path / Path("base_with_bubble") / Path(time_str)
        with_bubble_img_output_path.mkdir(parents=True)

        copy_bubble(src_path, base_path, mask_output_path, True, with_bubble_img_output_path, model_type, blend_method, detection_th)

        logger.info(f"Output : {with_bubble_img_output_path}")
        logger.info(f"Total Elapsed time : {time.time() - start_tim}")


fire.Fire(Command)
