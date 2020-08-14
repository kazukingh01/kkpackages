import numpy as np
import pandas as pd
import cv2
import os, sys
import glob
from functools import partial
from typing import List, Tuple
from joblib import Parallel, delayed
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


def convert_seg_point_to_bool(img_height: int, img_width: int, segmentations: List[List[float]]) -> np.ndarray:
    """
    Detectron2 では gt_mask では polygon形式であるが、pred_maskではbooleanで表現される
    polygon 形式 [x1, y1, x2, y2, ..] で囲まれた segmentation の area を boolean に変更する関数
    Params::
        img_height: int, 
        img_width:  int, 
        segmentations: [[x11,y11,x12,y12,...], [x21,y21,x22,y22,...], ...]. ※[seg1, seg2, ...]となっている
    """
    img = np.zeros((int(img_height), int(img_width), 3)).astype(np.uint8)
    img_add = img.copy()
    for seg in segmentations:
        # segmentation を 線を繋いで描く
        ndf = cv2.polylines(img.copy(), [np.array(seg).reshape(-1,1,2).astype(np.int32)], True, (255,255,255))
        # 線を描いた後、一番外の輪郭を取得する
        contours, _ = cv2.findContours(ndf[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 一番外の輪郭内部を埋める
        ndf = cv2.drawContours(ndf, contours, -1, (255,255,255), -1)
        img_add += ndf
    # boolrean に変更する
    img_add = (img_add[:, :, 0] > 0).astype(bool)
    return img_add


def compute_diameter(ndf: np.ndarray, preview: bool=False, resize: int=None):
    """
    diameter を計算したい mask (np.ndarray[bool]) を入力する
    Params::
        ndf: np.ndarray[bool]. ndf.shape = (縦, 横)
    ※補足::
        detectron2 の場合は次のように使う
        output["instances"].get("pred_masks")[ index_pole ].to("cpu").detach().numpy()
    """
    # mask の各座標を求める
    points = np.concatenate([[x.tolist()] for x in np.where(ndf)], axis=0) # 0:縦, 1:横
    # 1次関数で Fitting
    params = np.polyfit(points[1], points[0], 1) # cv2は左上rを原点としてy軸は下向きが生の方向なので反転する
    line1_y = lambda x: params[0] * x + params[1]
    line1_x = lambda y: (y - params[1]) / params[0]
    # input画像との交点を計算する
    y_w_min = line1_y(0)
    y_w_max = line1_y(ndf.shape[1])
    x_h_min = line1_x(0)
    x_h_max = line1_x(ndf.shape[0])
    x1, y1, x2, y2, bool_slope = 0, 0, 0, 0, False
    if   0 <= x_h_min and x_h_min <= ndf.shape[1] and 0 <= x_h_max and x_h_max <= ndf.shape[1]:
        x1, y1, x2, y2 = int(x_h_min), 0, int(x_h_max), ndf.shape[0]
        bool_slope = True # 線が、画像の上下に横断している場合
    elif 0 <= y_w_min and y_w_min <= ndf.shape[0] and 0 <= y_w_max and y_w_max <= ndf.shape[0]:
        x1, y1, x2, y2 = 0, int(y_w_min), ndf.shape[1], int(y_w_max)
    # 真っ黒な画像を作成
    img  = np.zeros((ndf.shape[0], ndf.shape[1], 3)).astype(np.uint8)
    # mask 領域に色を塗る
    img[:, :, 0][ndf] = 255
    img_mask = img.copy() # diameter 算出に使うのでここで一旦保持. 白画像にしておく
    img_mask[:, :, 1][ndf] = 255 
    img_mask[:, :, 2][ndf] = 255 
    # 線を引く
    img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1, lineType=cv2.LINE_8, shift=0) # 線を引く
    # Fittingした１次関数に直行する傾きを持つ１次関数を定義
    line2_b = lambda x, y: y + (1 / params[0] * x)
    # b(切片) の範囲を計算する
    b_min, b_max = 0, 0
    if bool_slope:
        # Fittingが画像の上下に横断している。つまり直行する線は画面の左右を横断
        b_min = line2_b(line1_x(0), 0)
        b_max = line2_b(line1_x(ndf.shape[0]), ndf.shape[0])
    else:
        b_min = line2_b(0, line1_y(0))
        b_max = line2_b(ndf.shape[1], line1_y(ndf.shape[1]))
    # 直行する線を100本描く. 100本とsegmentation の and領域の端と端を取得する
    list_length = []
    for b in np.arange(b_min, b_max, (b_max - b_min)/100):
        line2_y = lambda x: -1 / params[0] * x + b
        line2_x = lambda y: -1 * (y - b) * params[0]
        x1, y1, x2, y2 = 0, 0, 0, 0
        if bool_slope:
            # 直行する線は画面の左右を横断
            x1, y1, x2, y2 = 0, int(line2_y(0)), ndf.shape[1], int(line2_y(ndf.shape[1]))
        else:
            # 直行する線は画面の上下を横断
            x1, y1, x2, y2 = int(line2_x(0)), 0, int(line2_x(ndf.shape[0])), ndf.shape[0]
        # 線を引く
        img   = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1, lineType=cv2.LINE_8, shift=0) # 線を引く
        # 線を引いて、segmentation した領域との and 条件を撮る
        imgwk = cv2.line(np.zeros_like(img_mask).astype(np.uint8), (x1, y1), (x2, y2), (255, 255, 255), thickness=1, lineType=cv2.LINE_8, shift=0) # 線を引く
        imgwk = (imgwk[:, :, 0] > 0) & (img_mask[:, :, 0] > 0) # and 条件
        if imgwk.sum() > 0:
            points_wk = np.concatenate([[x.tolist()] for x in np.where(imgwk)], axis=0) # 0:縦, 1:横
            # 縦で(横でもいいが)ポイントが最小・最大となる２点を取得する
            y_min = points_wk[0].min()
            y_max = points_wk[0].max()
            x_min = points_wk[1][points_wk[0] == y_min].min()
            x_max = points_wk[1][points_wk[0] == y_max].max()
            list_length.append(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2))
    length = round(np.median(list_length), 1)
    img = cv2.putText(img, str(length), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA)
    if resize is not None:
        img = fit_resize(img, "y", scale=resize)
    if preview:
        cv2.imshow("test", img)
        cv2.waitKey(0)
    return length


def shape_fitting(
    ndf: np.ndarray, shape_polygon: List[float], 
    search_scale_x: (float, float) = (0.75, 1.001, 0.1), search_scale_y: (float, float) = (0.75, 1.001, 0.1),
    priority_type: str="r_target_and",
    calc_target_type: str="ratio", ascending: bool=True, n_calc: int=30, 
    preview: bool=False, n_jobs: int=1
) -> (np.ndarray, List[List[int]], pd.DataFrame, pd.Series):
    """
    入力された binary画像を元に、 binaryを回転させながら最適な短径を計算する
    Params::
        ndf: binary
        shape_polygon:
            x1, y1, t1, c1, ... cは次の点と繋がるかどうか。tはthickness, 辺1の正方形の中のどの位置に点を描くかをfloat で表す
            [0.2, 0.2, 1, 0.6, 0.2, 1, 0.8, 0.5, 1, 0.6, 0.8, 1, 0.2, 0.8, 0]
        search_scale_x:
            shape_polygon の形状を x方向にどの程度 scale して探索するか. 1.0倍のscaleはある短形の短い辺の長さが基準. np.arange()の中身
        search_scale_y:
            shape_polygon の形状を y方向にどの程度 scale して探索するか. 1.0倍のscaleはある短形の短い辺の長さが基準. np.arange()の中身
        priority_type:
            r_target_and or r_shape_and
        calc_target_type: ratio or area. rationは 短形の 長辺/短辺
        n_calc: calc_target_type をsortした時に、上からどれだけの候補を計算に使うか
        thickness: shape を記述する際の線の太さ
    Return::
        4つの変数が返却される
            1: 探索したい shape の mask
            2: 探索したい shape の keypoints. [[x1,y1], [x2,y2], ...]
            3: 計算した DataFrame
            4: 3のDataFrame内のBestなSeries
    """
    points   = np.concatenate([[x] for x in np.where(ndf > 0)[::-1]], axis=0)
    rotation = lambda x: np.array([np.cos(np.deg2rad(x)), -np.sin(np.deg2rad(x)), np.sin(np.deg2rad(x)), np.cos(np.deg2rad(x))]).reshape(2,2)
    # shape を1辺=100で作成する
    img_shape = np.zeros((100,100)).astype(np.uint8)
    ndf_shape = np.array(shape_polygon).reshape(-1, 4)
    for i in range(ndf_shape.shape[0] - 1):
        x1, y1, t1, c1, x2, y2, t2, c2 = ndf_shape[i:i+2].reshape(-1)
        x1, y1, x2, y2 = int(img_shape.shape[1]*x1), int(img_shape.shape[0]*y1), int(img_shape.shape[1]*x2), int(img_shape.shape[0]*y2)
        if c1 == 1:
            img_shape = cv2.line(img_shape, (x1, y1), (x2, y2), (255, 255, 255), thickness=int(t1), lineType=cv2.LINE_AA)
        if ndf_shape.shape[0] - 2 == i:
            if c2 == 1:
                img_shape = cv2.line(img_shape, (x2, y2), (int(ndf_shape[0][0]), int(ndf_shape[0][1])), (255, 255, 255), thickness=int(t2), lineType=cv2.LINE_AA)
    # 描画処理
    if preview:
        cv2.imshow("test", img_shape)
        cv2.waitKey(0)
    df_deg_area = pd.DataFrame()
    for deg in range(360):
        points_rot = rotation(deg) @ points # 回転させる
        points_rot[0] = points_rot[0] - points_rot[0].min()
        points_rot[1] = points_rot[1] - points_rot[1].min()
        x_max = int(points_rot[0].max()) + 1
        y_max = int(points_rot[1].max()) + 1
        area  = x_max * y_max # min は x,y共に0に補正している
        ratio = x_max / y_max if x_max > y_max else y_max / x_max
        df_deg_area = df_deg_area.append({"rotation": deg, "area":area, "ratio":ratio}, ignore_index=True, sort=False)

    # 並列計算用に関数化する
    def __work(deg: float, rotation=None, points: np.ndarray=None, img_shape: np.ndarray=None, search_scale_x: Tuple[float]=None, search_scale_y: Tuple[float]=None) -> pd.DataFrame:
        print(f"compute degree: {deg}")
        points_rot = rotation(deg) @ points # 回転させる
        x_min = points_rot[0].min()
        y_min = points_rot[1].min()
        x_max = points_rot[0].max()
        y_max = points_rot[1].max()
        ## mask画像を作成する
        ndfwk = np.zeros(( int(y_max - y_min)+1, int(x_max - x_min)+1 )).astype(bool)
        ndfwk[(points_rot[1] - y_min).astype(int), (points_rot[0] - x_min).astype(int)] = True
        # mask画像内にshape画像を探索する. scaleは0.5〜1.0まで
        list_values = []
        for y_scale in np.arange(*search_scale_y):
            for x_scale in np.arange(*search_scale_x):
                img_shape_wk = cv2.resize(img_shape.copy() , (int(min(ndfwk.shape) * x_scale), int(min(ndfwk.shape) * y_scale))) # 横, 縦
                if img_shape_wk.shape[0] > ndfwk.shape[0] or img_shape_wk.shape[1] > ndfwk.shape[1]: continue # scaleの結果探索サイズを超えたらcontinue
                img_shape_wk = (img_shape_wk > 0)
                ## mask画像に対してshape画像のFit具合を探索する
                for index_y in np.arange(0, ndfwk.shape[0] - img_shape_wk.shape[0]):
                    for index_x in np.arange(0, ndfwk.shape[1] - img_shape_wk.shape[1]):
                        n_and  = (ndfwk[index_y:index_y+img_shape_wk.shape[0], index_x:index_x+img_shape_wk.shape[1]] & img_shape_wk).sum()
                        r_shape_and  = n_and / img_shape_wk.sum()
                        r_target_and = n_and / ndfwk.sum()
                        list_values.append((deg, x_min, y_min, img_shape_wk.shape[1], img_shape_wk.shape[0], index_x, index_y, n_and, r_shape_and, r_target_and))
        # 結果を格納
        if len(list_values) == 0: return pd.DataFrame() #空DF
        ndf_values = np.array(list_values)
        dfwk = pd.DataFrame(ndf_values, columns=["rotation", "base_x", "base_y", "x_scale", "y_scale", "index_x", "index_y", "n_and", "r_shape_and", "r_target_and"])
        return dfwk
    ## partial で変数を埋め込む
    func = partial(__work, rotation=rotation, points=points, img_shape=img_shape, search_scale_x=search_scale_x, search_scale_y=search_scale_y)
    # area の min から順に30個計算する(並列計算する)
    list_degs = df_deg_area.sort_values(calc_target_type, ascending=ascending)["rotation"].values[:n_calc].tolist()
    list_df = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)([delayed(func)(x) for x in list_degs])
    # 結果を結合
    df = pd.concat(list_df, axis=0, ignore_index=True, sort=False)
    sewk = df[df[priority_type] == df[priority_type].max()].iloc[0] #priority_typeの最大で判断する
    # 一番Fitした画像を使って、元画像sizeに合わせたmaskを作成する
    img_shape_wk = cv2.resize(img_shape.copy() , (int(sewk["x_scale"]), int(sewk["y_scale"]))) # 横, 縦
    points_shape = np.concatenate([[x] for x in np.where(img_shape_wk > 0)[::-1]], axis=0).astype(float) # x,y
    points_shape[0] = points_shape[0] + sewk["base_x"] + sewk["index_x"]
    points_shape[1] = points_shape[1] + sewk["base_y"] + sewk["index_y"]
    points_shape = rotation(-sewk["rotation"]) @ points_shape # shape mask画像のポイントを反rotationする
    points_shape = points_shape.astype(int)
    ## points_shape が回転の結果index overしている事があるのでそういった点は外す
    points_shape = points_shape[:, ~(points_shape[0] < 0) & ~(points_shape[0] >= ndf.shape[1]) & ~(points_shape[1] < 0) & ~(points_shape[1] >= ndf.shape[0])]
    img_mask = np.zeros(ndf.shape).astype(bool)
    img_mask[points_shape[1], points_shape[0]] = True
    # Fit元のobjectのKeypointを元画像sizeに合わせたpointを作成する
    shape_keypoints = []
    for i in range(ndf_shape.shape[0]):
        x1, y1, _, _, = ndf_shape[i]
        x1 = x1 * img_shape_wk.shape[1] + sewk["base_x"] + sewk["index_x"]
        y1 = y1 * img_shape_wk.shape[0] + sewk["base_y"] + sewk["index_y"]
        x1, y1 = (rotation(-sewk["rotation"]) @ np.array((x1, y1)).reshape(2, -1)).reshape(-1)
        shape_keypoints.append([int(x1), int(y1)])
    # 描画処理
    if preview:
        img = np.zeros((*ndf.shape, 3)).astype(np.uint8)
        img[:, :, 0][ndf] = 255
        img[points_shape[1], points_shape[0], 0] = 0
        img[points_shape[1], points_shape[0], 1] = 255
        for x1, y1 in shape_keypoints:
            img = cv2.circle(img, (x1, y1), 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        img = fit_resize(img, "y", 900)
        cv2.imshow("test", img)
        cv2.waitKey(0)
    return img_mask, shape_keypoints, df, sewk


def compute_cross_point(p1: (float, float), p2: (float, float), target: List[float]) -> (float, float):
    """
    あるbase_lineとtarget(line or point)との交点を求める.
    targetがpointの場合は直行する線との交点とする
    Return::
        cross point x, cross point y
    Params::
        p1: base line の１点. x, y
        p2: base line の１点. x, y
        target: １点か２点を記述する。１点の場合は点、２点の場合は線とみなす
            [x1, y1] or [x1, y1, x2, y2]
    """
    bool_line = None
    if   len(target) == 2:
        bool_line = False
    elif len(target) == 4:
        bool_line = True
    else:
        raise Exception(f"target length must be 2 or 4. this target length is {len(target)}")
    cp_x, cp_y = None, None
    if p2[0] - p1[0] != 0:
        a1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - (a1 * p1[0])
        y1 = lambda x: a1 * x + b1
        if bool_line:
            if target[2] - target[0] != 0:
                a2 = (target[3] - target[1]) / (target[2] - target[0])
                b2 = target[1] - (a2 * target[0])
                cp_x = (b2 - b1)/(a1 - a2)
                cp_y = y1(cp_x)
            else:
                cp_x = target[0]
                cp_y = y1(cp_x)
        else:
            if a1 == 0:
                cp_x = target[0]
                cp_y = p1[1]
            else:
                a2 = -1/a1
                b2 = target[1] - (a2 * target[0])
                cp_x = (b2 - b1)/(a1 - a2)
                cp_y = y1(cp_x)
    else:
        if bool_line:
            if target[2] - target[0] != 0:
                a2 = (target[3] - target[1]) / (target[2] - target[0])
                b2 = target[1] - (a2 * target[0])
                cp_x = p1[0]
                cp_y = a2 * cp_x + b2
            else:
                raise Exception("The two lines are perfectly parallel.")
        else:
            cp_x = p1[0]
            cp_y = target[1]
    return cp_x, cp_y


def compute_iou(bbox1: (float, float, float, float), bbox2: (float, float, float, float)):
    """
    ２つのBBoxからIoUを計算する
    Params::
        bbox1: x1, y1, x2, y2
        bbox2: x1, y1, x2, y2
    """
    b1x1, b1y1, b1x2, b1y2 = bbox1
    b2x1, b2y1, b2x2, b2y2 = bbox2
    x_min = np.array([b1x1, b1x2, b2x1, b2x2]).min()
    x_max = np.array([b1x1, b1x2, b2x1, b2x2]).max()
    y_min = np.array([b1y1, b1y2, b2y1, b2y2]).min()
    y_max = np.array([b1y1, b1y2, b2y1, b2y2]).max()
    # 最小値で補正する. IoU計算する領域を少なくするため
    b1x1, b1x2, b2x1, b2x2 = int(b1x1 - x_min), int(b1x2 - x_min), int(b2x1 - x_min), int(b2x2 - x_min)
    b1y1, b1y2, b2y1, b2y2 = int(b1y1 - y_min), int(b1y2 - y_min), int(b2y1 - y_min), int(b2y2 - y_min)
    x_max, y_max = int(x_max - x_min), int(y_max - y_min)
    # bbox の大小が逆の場合は補正する
    b1x1, b1x2 = (b1x1, b1x2, ) if b1x1 < b1x2 else (b1x2, b1x1, )
    b1y1, b1y2 = (b1y1, b1y2, ) if b1y1 < b1y2 else (b1y2, b1y1, )
    b2x1, b2x2 = (b2x1, b2x2, ) if b2x1 < b2x2 else (b2x2, b2x1, )
    b2y1, b2y2 = (b2y1, b2y2, ) if b2y1 < b2y2 else (b2y2, b2y1, )
    # np.ndarray の and 領域でIoUを計算する
    img_binary  = np.zeros((y_max+1, x_max+1)).astype(bool)
    img_binary1 = img_binary.copy()
    img_binary1[b1y1:b1y2, b1x1:b1x2] = True
    img_binary2 = img_binary.copy()
    img_binary2[b2y1:b2y2, b2x1:b2x2] = True
    iou = (img_binary1 & img_binary2).sum() /  (img_binary1 | img_binary2).sum()
    return iou


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


def same_images_concat(dir_paths: List[str], save_images_path: str = None, str_regex: List[str]=[r"png$", r"jpg$"]):
    dict_ret = {}
    # 最初のdirectory を BASE にする
    list_images = get_file_list(dir_paths[0], str_regex)
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
            height_after = int(height * (scale / width))
        elif dim == "y":
            height_after = int(scale)
            width_after  = int(width * (scale / height))
    else:
        raise Exception(f"scale > 10.")
    img = cv2.resize(img , (width_after, height_after)) # 横, 縦
    return img


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

