import logging
import time
from pathlib import Path
import fire
from datetime import datetime
import shutil

from bubble_tool.bubble_tool import remove_bubble, copy_bubble, split_panel, combine_panel

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
        import os
        if os.name == 'nt':
            import _locale
            if not hasattr(_locale, '_gdl_bak'):
                _locale._gdl_bak = _locale._getdefaultlocale
                _locale._getdefaultlocale = (lambda *args: (_locale._gdl_bak()[0], 'UTF-8'))

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


    def copy(self, src_path, base_path, mask_path=None, model_type:int=0, blend_method:int=-1, detection_th:float=0.1):
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


    def split(self, src_path, frag_size=1024*2):
        # src image -> split images
        start_tim = time.time()

        src_path = Path(src_path)
        if not src_path.is_dir():
            raise ValueError( f"{src_path} not found" )        

        output_dir = Path("output") / Path("split") / Path(get_time_str())
        output_dir.mkdir(parents=True)

        split_panel(src_path, Path(output_dir), frag_size)

        logger.info(f"Output : {output_dir}")
        logger.info(f"Total Elapsed time : {time.time() - start_tim}")


    def combine(self, src_path, frag_path):
        # src image + frag images -> combined image
        start_tim = time.time()

        src_path = Path(src_path)
        if not src_path.is_dir():
            raise ValueError( f"{src_path} not found" )        

        frag_path = Path(frag_path)
        if not frag_path.is_dir():
            raise ValueError( f"{frag_path} not found" )
        
        output_dir = Path("output") / Path("combine") / Path(get_time_str())
        output_dir.mkdir(parents=True)

        combine_panel(src_path, frag_path, Path(output_dir))

        logger.info(f"Output : {output_dir}")
        logger.info(f"Total Elapsed time : {time.time() - start_tim}")


    def remove_proj(self, proj_dir_path, model_type:int=0, detection_th:float=0.1, split=True, frag_size=1024*2):
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

        if split == False:
            clean_img_output_path = proj_dir_path / Path("cleaned") / Path(time_str)
            clean_img_output_path.mkdir(parents=True)
            remove_bubble(src_path, mask_output_path, clean_img_output_path, model_type, detection_th)

            logger.info(f"Output : {clean_img_output_path}")
        else:
            clean_img_output_path = proj_dir_path / Path("pre_split") / Path(time_str)
            clean_img_output_path.mkdir(parents=True)
            split_output_path = proj_dir_path / Path("cleaned") / Path(time_str)
            split_output_path.mkdir(parents=True)
            remove_bubble(src_path, mask_output_path, clean_img_output_path, model_type, detection_th)

            split_panel(clean_img_output_path, split_output_path, frag_size)

            split_info_path = split_output_path / Path("split_info.txt")
            split_info_path2 = proj_dir_path / Path("base") / Path("split_info.txt")

            shutil.copy(split_info_path, split_info_path2)

            logger.info(f"Output : {split_output_path}")

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")


    def copy_proj(self, proj_dir_path, model_type:int=0, blend_method:int=-1, detection_th:float=0.1):
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

        combine = (base_path / Path("split_info.txt")).is_file()

        if combine == False:
            copy_bubble(src_path, base_path, mask_output_path, True, with_bubble_img_output_path, model_type, blend_method, detection_th)
        else:
            combine_output_path = proj_dir_path / Path("combined") / Path(time_str)
            combine_output_path.mkdir(parents=True)

            combine_panel(src_path, base_path, combine_output_path)
            copy_bubble(src_path, combine_output_path, mask_output_path, True, with_bubble_img_output_path, model_type, blend_method, detection_th)


        logger.info(f"Output : {with_bubble_img_output_path}")
        logger.info(f"Total Elapsed time : {time.time() - start_tim}")


fire.Fire(Command)
