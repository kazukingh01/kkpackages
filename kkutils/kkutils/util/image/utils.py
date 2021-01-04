import cv2
import numpy as np
from PIL import Image


__all__ = [
    "fit_resize",
    "add_image",
    "pil2cv",
    "cv2pil",
]


def fit_resize(img: np.ndarray, dim: str, scale: int):
    """
    Params::
        img: image
        dim: x or y
        scale: width or height
    """
    if dim not in ["x","y"]: raise Exception(f"dim: {dim} is 'x' or 'y'.")
    # scale の 数値に合わせて resize する
    height = img.shape[0]
    width  = img.shape[1]
    height_after, width_after = None, None
    if   type(scale) == int and scale > 10:
        if   dim == "x":
            width_after  = int(scale)
            height_after = int(height * (scale / width))
        elif dim == "y":
            height_after = int(scale)
            width_after  = int(width * (scale / height))
    else:
        raise Exception(f"scale > 10.")
    img = cv2.resize(img , (width_after, height_after)) # 横, 縦
    return img

def add_image(img: np.ndarray, img_add: np.ndarray, loc: (int, int), color_trans: (int, int, int)=None):
    """
    Params::
        img: image
        img_add: add image
        loc: location. (x, y)
    """
    h, w = img_add.shape[:2]
    if color_trans is None:
        img[loc[1]:loc[1]+h, loc[0]:loc[0]+w] = img_add.copy()
    else:
        if len(color_trans) != img.shape[-1]: raise Exception(f"No match dimension. color_trans: {color_trans}, shape: {img.shape}")
        imgwk = np.zeros_like(img).astype(np.uint8)
        imgwk[:] = color_trans
        imgwk[loc[1]:loc[1]+h, loc[0]:loc[0]+w] = img_add.copy()
        mask  = np.ones(imgwk.shape[:2]).astype(np.bool)
        masks = (imgwk == color_trans)
        for i in range(masks.shape[-1]): mask = mask & masks[:, :, i]
        img[~mask] = imgwk[~mask]
    return img

def pil2cv(img: Image) -> np.ndarray:
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(img, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(img: np.ndarray):
    ''' OpenCV型 -> PIL型 '''
    new_image = img.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image