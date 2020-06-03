import numpy as np
import cv2
import os, sys
import glob
from typing import List, Tuple

# local lib
from kkimagemods.util.common import check_type, get_file_list, correct_dirpath, makedirs

def drow_bboxes(img_org: np.ndarray, bboxes: List[List[float]], bboxes_class: List[str]=None, bbox_type="wh", color=(0,255,0), thickness: int=1, font_scale: float=1.0):
    """
    bbox を加えて描画していく

    Params
    ----------------
    img_org:
        書き込む元の画像
    bboxes:
        [x, y, w, h] or [[x, y, w, h], [x, y, w, h], ...]
    bbox_type: wh or xy
    """
    def __work(x1, y1, x2, y2, bbox_type: str):
        if   bbox_type == "xy":
            return (int(x1),int(y1),), (int(x2),int(y2),)
        elif bbox_type == "wh":
            return (int(x1),int(y1),), (int(x1+x2),int(y1+y2),)

    img = img_org.copy()
    if len(bboxes) == 4 and type(bboxes[0]) != list:
        # bbox が１つの場合
        p1, p2 = __work(*bboxes, bbox_type)
        img = cv2.rectangle(img, p1, p2 ,color, thickness=thickness)
        if bboxes_class is not None:
            img = cv2.putText(img, bboxes_class, p1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness=thickness)
    else:
        # bbox が複数の場合
        for i, bbox in enumerate(bboxes):
            p1, p2 = __work(*bbox, bbox_type)
            img = cv2.rectangle(img, *__work(*bbox, bbox_type) ,color, thickness=thickness)
            if bboxes_class is not None:
                img = cv2.putText(img, bboxes_class[i], p1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness=thickness)
    return img


def add_image_in_region(binary: np.ndarray, addImage: np.ndarray, \
    x: int, y: int, width: int, height: int, extend_width: int=0) -> np.ndarray:
    """
    addImageの中のx,y,x+width,y+heightにある領域の中だけをbinaryに足す
    """
    _binary = binary.copy().astype(np.int32)

    # 長方形の中の edge の情報を書き加える(+aさらに広げる)
    y1 = y - extend_width
    y2 = y + height + 1 + extend_width
    y1 = y1 if y1 >= 0 else 0
    y2 = y2 if y2 < addImage.shape[0] else addImage.shape[0]
    x1 = x - extend_width
    x2 = x + width + 1 + extend_width
    x1 = x1 if x1 >= 0 else 0
    x2 = x2 if x2 < addImage.shape[1] else addImage.shape[1]

    # 該当範囲以外の領域をゼロで埋める
    ndf = addImage.copy()
    ndf[:y1, :] = 0
    ndf[y2:, :] = 0
    ndf[:, :x1] = 0
    ndf[:, x2:] = 0

    # uint8 に変換する
    _binary += ndf.astype(np.int32)
    _binary = np.uint8(np.clip(_binary,0,255))
    return _binary 


def draw_bounding_box(binary: np.ndarray, threshold_boxsize: int=None, \
    color: (int, int, int)=(0, 255, 0), thickness: int=2) -> np.ndarray:
    """
    binary の画像に bounding box を見つけて、bounding boxの枠のみを返却する
    """
    result = np.zeros(list(binary.shape) + [3]).astype(np.uint8)

    # 輪郭を発見する
    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    # 輪郭のサイズの閾値が定義されていれば適用する
    if threshold_boxsize is not None and type(threshold_boxsize) == int:
        contours = list(filter(lambda x: cv2.contourArea(x) > threshold_boxsize, contours))

    # 輪郭を矩形で囲む。
    for i, cnt in enumerate(contours):
        # 輪郭に外接する長方形を取得する
        x, y, width, height = cv2.boundingRect(cnt)
        # 描画する
        result = cv2.rectangle(result, (x, y), (x + width, y + height), \
                               color=color, thickness=thickness)
    
    # 短形のみを返却する
    return result
    

def dense_optical_flow(gray_prev: np.ndarray, gray_next: np.ndarray, mask_lower: int = 0, mask_thre: int = 0,
                       pyr_scale: float = 0.5, levels: int = 3, winsize: int = 10, iterations: int = 5, 
                       poly_n: int = 5, poly_sigma: float = 1.1, flags: int = 0) -> np.ndarray:
    ## Dense Optical Flow
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_next, None,
                pyr_scale = pyr_scale, levels = levels, winsize = winsize, iterations = iterations, 
                poly_n = poly_n, poly_sigma = poly_sigma, flags = flags
    )

    # dense optical flow の結果
    mag, _ = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True) # 第二出力はangle

    # 結果を書くためのmaskを用意
    mask = np.zeros(list(gray_next.shape) + [3]).astype(np.uint8)
    mask[:, :, 0] = mask[:, :, 1] = 0
    mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 閾値
    if mask_thre > 0:
        mask[:, :, 2][mask[:,:,2] < mask_thre] = 0

    # 変化がわかりづらいので色のrangeの最低値を上げる
    if mask_lower > 0:
        mask[:, :, 2][mask[:, :, 2] > 0] = (((mask[:, :, 2][mask[:, :, 2] > 0]).astype(np.float32) * ((255-mask_lower)/255)) + mask_lower).astype(np.uint8)
    
    return mask


def morphology(gray: np.ndarray, small_kernel: np.ndarray = np.ones((5,5),np.uint8), 
    large_kernel: np.ndarray = np.ones((15,15),np.uint8)) -> np.ndarray:

    binary = cv2.inRange(gray, (0, 0, 1), (0, 0, 255))
    binary = cv2.erode( binary, small_kernel, iterations = 3)
    binary = cv2.dilate(binary, small_kernel, iterations = 3)
    binary = cv2.dilate(binary, small_kernel, iterations = 3)
    binary = cv2.erode( binary, small_kernel, iterations = 3)

    return binary


def add_frame_grayscale(inputImage:np.ndarray, addImage:np.ndarray) -> np.ndarray:
    return np.clip((inputImage.astype(int) + addImage.astype(int)), 0, 255).astype(np.uint8)


def make_collage(input_dirpath: str, str_regex: List[str], size: (int, int) = (1920, 1080), n_images_y : int = 5):
    """
    １枚のコラージュ画像作成関数
    指定したsizeの縦に何枚の画像を埋めるか(n_images_y)に従って縦をresizeして横に連連結し、１枚の画像を作っていく
    """
    # collage 画像を作るためのinput 画像を取得
    _input_dirpath = input_dirpath if input_dirpath[-1] == "/" else input_dirpath + "/"
    files = []
    for x in str_regex: files += glob.glob(_input_dirpath + x)

    # random に画像を取得してresizeして画像を作成する
    img_out, i_y = [None] * n_images_y, 0
    for i_rand in np.random.permutation(np.arange(len(files))):
        img = cv2.imread(files[i_rand]) # img.shape = (640, 480, 3) 縦,横,カラー
        img = cv2.resize(img , (int(img.shape[1]*((size[1]//n_images_y + 1)/img.shape[0])), int(size[1]//n_images_y + 1))) # 横, 縦
        if img_out[i_y] is None:
            ## 初回は copy
            img_out[i_y] = img.copy()
        else:
            ## ２回目以降は横に連結
            img_out[i_y] = cv2.hconcat([img_out[i_y], img])
        
        # img_outの各要素のimg の幅が size[0] を超えたら次の要素に
        if img_out[i_y].shape[1] > size[0]:
            # 横画像の幅を揃えるためにsizeで打ち切り
            img_out[i_y] = img_out[i_y][:, :size[0], :]
            i_y += 1
        if i_y >= n_images_y: break
    
    # 最後に縦に連結
    img_out = cv2.vconcat(img_out)
    img_out = img_out[:size[1], :, :]

    return img_out


def output_collages(output_dirpath: str, input_dirpath: str, str_regex: List[str], n_ouptuts: int, *args, **kwargs):
    _output_dirpath = output_dirpath if output_dirpath[-1] == "/" else output_dirpath + "/"
    for i in range(n_ouptuts):
        print(f"output :{i}")
        img = make_collage(input_dirpath, str_regex, *args, **kwargs)
        cv2.imwrite(_output_dirpath + str(i) + ".jpg", img)


def concats(file_paths: List[str]):
    """
    file_paths にある画像を横に連結させる
    file_paths に画像がなかった場合は先頭の画像をbaseに空画像を挟む
    """
    img_ret, img_zero, height = None, None, None
    for x in file_paths:
        img = cv2.imread(x)
        if img_ret is None:
            img_ret   = img.copy()
            img_zero  = np.zeros_like(img)
            height    = img.shape[0]
        elif img is None:
            img_ret = cv2.hconcat([img_ret, img_zero])
        else:
            img = fit_resize(img, "y", height)
            img_ret = cv2.hconcat([img_ret, img])
    return img_ret


def same_images_concat(dir_paths: List[str], save_images_path: str = None):
    dict_ret = {}
    # 最初のdirectory を BASE にする
    list_images = get_file_list(dir_paths[0], ["png$", "jpg$"])
    for x in [os.path.basename(y) for y in list_images]:
        dict_ret[x] = concats([correct_dirpath(_x) + x for _x in dir_paths])

    if save_images_path is not None:
        makedirs(save_images_path, exist_ok=True, remake=False)
        for x in dict_ret.keys():
            if save_images_path is not None:
                cv2.imwrite(correct_dirpath(save_images_path) + x, dict_ret[x])

    return dict_ret
    


def fit_resize(img: np.ndarray, dim: str, scale):
    check_type(scale, [int])
    if dim not in ["x","y"]: raise Exception(f"dim: {dim} is 'x' or 'y'.")

    # scale の 数値に合わせて resize する
    height = img.shape[0]
    width  = img.shape[1]
    height_after, width_after = None, None
    if   type(scale) == int and scale > 10:
        if   dim == "x":
            width_after  = int(scale)
            height_after = int(height * (width / scale))
        elif dim == "y":
            height_after = int(scale)
            width_after  = int(width * (height / scale))
    else:
        raise Exception(f"scale > 10.")
    img = cv2.resize(img , (width_after, height_after)) # 横, 縦
    return img


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

