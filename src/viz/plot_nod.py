import os
import os.path as op
import math
from PIL import Image

def get_img_path(path_root:str,
                 imgid:str
                 ) -> str:
    """Get image path of NOD.
    
    Parameters
    ----------
    path_root : str
        Root path of the image.
    imgid : str
        Image ID of NOD image.
    
    Returns
    -------
    str
        Image path.
    """
    task = 'ImageNet' if 'n' in imgid else 'CoCo'
    suffix = 'JPEG' if 'n' in imgid else 'jpg'
    img = imgid.zfill(12) if task == 'CoCo' else imgid
    return op.join(path_root,task,f'{img}.{suffix}')

def get_img_id(path):
    return path.split('/')[-1].split('.')[0]

def combine_images(image_ids: list[str],
                   stim_root: str,
                   ) -> Image:
    """Combine images.
    
    Parameters
    ----------
    image_ids : list[str]
        Image IDs.
    stim_root : str
        Root path of the image.
    
    Returns
    -------
    Image
        Combined image.
    """
    image_paths = [
        get_img_path(stim_root, imgid) for imgid in image_ids
        ]
    
    num_images = len(image_paths)
    grid_size = math.ceil(math.sqrt(num_images))
    num_rows = grid_size
    num_cols = grid_size if num_images > (
        grid_size - 1
        ) * grid_size else grid_size - 1

    images = [
        Image.open(img_path) for img_path in image_paths
        ]
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    combined_image = Image.new(
        'RGB', 
        (max_width * num_cols, max_height * num_rows)
        )

    for index, img in enumerate(images):
        row = index // num_cols
        col = index % num_cols
        combined_image.paste(
            img, (col * max_width, row * max_height)
            )

    return combined_image