import torch
import numpy as np


__all__ = [
    "corr_coef",
    "corr_coef_2ndarray"
]


def corr_coef(_ndf: np.ndarray, _dtype: str="float16", min_n_not_nan: int=10) -> np.ndarray:
    """
    相関係数の計算をGPU化.
    """
    _dtype = getattr(torch, _dtype)
    # float16だと内積の計算で発散するので先に規格化する
    with torch.no_grad():
        """
        >>> _ndf
        array([[ 2.,  1.,  3., nan, nan],
               [ 0.,  2.,  2., nan, -3.],
               [ 8.,  2.,  9., nan, -7.],
               [ 1., nan,  5.,  5., -1.]])
        """
        tensor_max = torch.from_numpy(np.nanmax(_ndf, axis=0)).to(_dtype).to("cuda:0")
        tensor_min = torch.from_numpy(np.nanmin(_ndf, axis=0)).to(_dtype).to("cuda:0")
        tensor_max = (tensor_max - tensor_min).cpu().numpy()
        tensor_max[tensor_max == 0] = float("inf") # 0除算を避けるため
        tensor_max = torch.from_numpy(tensor_max).to("cuda:0")
        """
        >>> tensor_min
        tensor([ 0.,  1.,  2.,  5., -7.], device='cuda:0', dtype=torch.float16)
        >>> tensor_max
        tensor([8., 1., 7., inf, 6.], device='cuda:0', dtype=torch.float16)
        """
        ndf = torch.from_numpy(_ndf).to(_dtype).to("cuda:0")
        ndf = (ndf - tensor_min) / tensor_max
        del tensor_min, tensor_max
        # 列ペア毎のmeanを計算する
        is_nan = torch.isnan(ndf)
        """
        >>> is_nan
        tensor([[False, False, False,  True,  True],
                [False, False, False,  True, False],
                [False, False, False,  True, False],
                [False,  True, False, False, False]], device='cuda:0')
        """
        ndf = ndf.cpu()
        ndf[is_nan.cpu()] = 0
        ndf = ndf.to(_dtype).to("cuda:0")
        """
        >>> ndf
        tensor([[0.2500, 0.0000, 0.1428, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.0000, 0.0000, 0.6665],
                [1.0000, 1.0000, 1.0000, 0.0000, 0.0000],
                [0.1250, 0.0000, 0.4285, 0.0000, 1.0000]], device='cuda:0', dtype=torch.float16)
        """
        tensor_sum = torch.mm((~is_nan).to(_dtype).t(), ndf).to(_dtype)
        """
        >>> tensor_sum
        tensor([[1.3750, 2.0000, 1.5713, 0.0000, 1.6660],
                [1.2500, 2.0000, 1.1426, 0.0000, 0.6665],
                [1.3750, 2.0000, 1.5713, 0.0000, 1.6660],
                [0.1250, 0.0000, 0.4285, 0.0000, 1.0000],
                [1.1250, 2.0000, 1.4287, 0.0000, 1.6660]], device='cuda:0', dtype=torch.float16)
        """
        n_not_nan = torch.mm((~is_nan).t().to(_dtype), (~is_nan).to(_dtype)).to(_dtype)
        n_not_nan = n_not_nan.cpu().numpy()
        n_not_nan[(n_not_nan < float(min_n_not_nan))] = float("inf")
        n_not_nan = torch.from_numpy(n_not_nan).to(_dtype).to("cuda:0")
        """
        >>> n_not_nan
        tensor([[4., 3., 4., 1., 3.],
                [3., 3., 3., 0., 2.],
                [4., 3., 4., 1., 3.],
                [1., 0., 1., 1., 1.],
                [3., 2., 3., 1., 3.]], device='cuda:0', dtype=torch.float16)
        """
        tensor_mean = tensor_sum / n_not_nan # 列ペア毎のmean
        """
        >>> tensor_mean
        tensor([[0.3438, 0.6665, 0.3928, 0.0000, 0.5552],
                [0.4167, 0.6665, 0.3809,    nan, 0.3333],
                [0.3438, 0.6665, 0.3928, 0.0000, 0.5552],
                [0.1250,    nan, 0.4285, 0.0000, 1.0000],
                [0.3750, 1.0000, 0.4763, 0.0000, 0.5552]], device='cuda:0', dtype=torch.float16)
        """
        # 共分散を計算する
        """
        1/n * {Sigma(xi * yi) - Sigma(xi * ym) - Sigma(yi * xm) + Sigma(xm * ym)}
        1/n * {Sigma(xi * yi) - ym*Sigma(xi) - xm*Sigma(yi) + n * xm * ym}
        """
        tensor_xiyi = torch.mm(ndf.t(), ndf)
        tensor_xiym =  tensor_mean.t() * tensor_sum
        tensor_yixm = (tensor_mean.t() * tensor_sum).t()
        tensor_xmym =  tensor_mean.t() * tensor_mean * n_not_nan
        tensor_Sxy  = (tensor_xiyi - tensor_xiym - tensor_yixm + tensor_xmym) / n_not_nan
        """
        >>> tensor_xiyi
        tensor([[1.0781, 1.0000, 1.0889, 0.0000, 0.1250],
                [1.0000, 2.0000, 1.0000, 0.0000, 0.6665],
                [1.0889, 1.0000, 1.2041, 0.0000, 0.4285],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.1250, 0.6665, 0.4285, 0.0000, 1.4443]], device='cuda:0', dtype=torch.float16)
        >>> tensor_xiym
        tensor([[0.4727, 0.8335, 0.5400, 0.0000, 0.6250],
                [0.8330, 1.3330, 0.7617,    nan, 0.6665],
                [0.5400, 0.7617, 0.6172, 0.0000, 0.7935],
                [0.0000,    nan, 0.0000, 0.0000, 0.0000],
                [0.6245, 0.6665, 0.7930, 0.0000, 0.9248]], device='cuda:0', dtype=torch.float16)
        >>> tensor_Sxy
        tensor([[ 0.1514,  0.0557,  0.1372,  0.0000, -0.1666],
                [ 0.0557,  0.2223,  0.0794,     nan,  0.0000],
                [ 0.1372,  0.0794,  0.1467,  0.0000, -0.1218],
                [ 0.0000,     nan,  0.0000,  0.0000,  0.0000],
                [-0.1666,  0.0000, -0.1218,  0.0000,  0.1730]], device='cuda:0', dtype=torch.float16)
        """
        """
        1/n * {Sigma(xi * xi) - 2xm*Sigma(xi) + n * xm * xm}
        """
        tensor_xi2  = ndf * ndf
        tensor_xi2  = torch.mm((~is_nan).to(_dtype).t(), tensor_xi2).to(_dtype)
        tensor_Sxx  = (tensor_xi2 - 2 * (tensor_mean * tensor_sum) + tensor_mean ** 2 *  n_not_nan) / n_not_nan
        tensor_Syy  = tensor_Sxx.t()
        # numpyに戻すと少し計算がずれる(infがあればnanにしておく)
        ndf = (tensor_Sxy / torch.sqrt(tensor_Sxx * tensor_Syy)).to(_dtype).cpu().numpy()
        ndf[ndf ==  np.inf] = np.nan
        ndf[ndf == -np.inf] = np.nan
    return ndf


def corr_coef_2ndarray(_ndf1: np.ndarray, _ndf2: np.ndarray, _dtype: str="float16", min_n_not_nan: int=10) -> np.ndarray:
    """
    相関係数の計算をGPU化. 別のarrayでの計算
    """
    _dtype = getattr(torch, _dtype)
    with torch.no_grad():
        """
        >>> _ndf1
        array([[ 2.,  1.,  3., nan, nan],
            [ 0.,  2.,  2., nan, -3.],
            [ 8.,  2.,  9., nan, -7.],
            [ 1., nan,  5.,  5., -1.]])
        >>> _ndf2
        array([[ 2.,  1., nan],
            [ 2., nan, -3.],
            [ 3., -2., -1.],
            [ 9., nan,  3.]])
        """
        ndf1, ndf2 = None, None
        for i, _ndf in enumerate([_ndf1, _ndf2]):
            # Float16 でも値が overflow しないように規格化する
            tensor_max = torch.from_numpy(np.nanmax(_ndf, axis=0)).to(_dtype).to("cuda:0")
            tensor_min = torch.from_numpy(np.nanmin(_ndf, axis=0)).to(_dtype).to("cuda:0")
            tensor_max = (tensor_max - tensor_min).cpu().numpy()
            tensor_max[tensor_max == 0] = float("inf") # 0除算を避けるため
            tensor_max = torch.from_numpy(tensor_max).to("cuda:0")
            if i == 0:
                ndf1 = torch.from_numpy(_ndf).to(_dtype).to("cuda:0")
                ndf1 = (ndf1 - tensor_min) / tensor_max
            else:
                ndf2 = torch.from_numpy(_ndf).to(_dtype).to("cuda:0")
                ndf2 = (ndf2 - tensor_min) / tensor_max
            del tensor_min, tensor_max
        # 列ペア毎のmeanを計算する
        is_nan1 = torch.isnan(ndf1)
        is_nan2 = torch.isnan(ndf2)
        """
        >>> is_nan1
        tensor([[False, False, False,  True,  True],
                [False, False, False,  True, False],
                [False, False, False,  True, False],
                [False,  True, False, False, False]], device='cuda:0')
        """
        ndf1 = ndf1.cpu()
        ndf1[is_nan1.cpu()] = 0
        ndf1 = ndf1.to(_dtype).to("cuda:0")
        ndf2 = ndf2.cpu()
        ndf2[is_nan2.cpu()] = 0
        ndf2 = ndf2.to(_dtype).to("cuda:0")
        """
        >>> ndf1
        tensor([[0.2500, 0.0000, 0.1428, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.0000, 0.0000, 0.6665],
                [1.0000, 1.0000, 1.0000, 0.0000, 0.0000],
                [0.1250, 0.0000, 0.4285, 0.0000, 1.0000]], device='cuda:0', dtype=torch.float16)
        >>> ndf2
        tensor([[0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [0.1428, 0.0000, 0.3333],
                [1.0000, 0.0000, 1.0000]], device='cuda:0', dtype=torch.float16)
        """
        tensor_sum1 = torch.mm((~is_nan2).to(_dtype).t(), ndf1).to(_dtype)
        tensor_sum2 = torch.mm(ndf2.t(), (~is_nan1).to(_dtype)).to(_dtype)
        """
        >>> tensor_sum1
        tensor([[1.3750, 2.0000, 1.5713, 0.0000, 1.6660],
                [1.2500, 1.0000, 1.1426, 0.0000, 0.0000],
                [1.1250, 2.0000, 1.4287, 0.0000, 1.6660]], device='cuda:0', dtype=torch.float16)
        >>> tensor_sum2
        tensor([[1.1426, 0.1428, 1.1426, 1.0000, 1.1426],
                [1.0000, 1.0000, 1.0000, 0.0000, 0.0000],
                [1.3330, 0.3333, 1.3330, 1.0000, 1.3330]], device='cuda:0', dtype=torch.float16)
        """
        n_not_nan = torch.mm((~is_nan2).t().to(_dtype), (~is_nan1).to(_dtype)).to(_dtype)
        n_not_nan = n_not_nan.cpu().numpy()
        n_not_nan[(n_not_nan < float(min_n_not_nan))] = float("inf")
        n_not_nan = torch.from_numpy(n_not_nan).to(_dtype).to("cuda:0")
        """
        >>> n_not_nan
        tensor([[4., 3., 4., 1., 3.],
                [2., 2., 2., 0., 1.],
                [3., 2., 3., 1., 3.]], device='cuda:0', dtype=torch.float16)
        """
        tensor_mean1 = tensor_sum1 / n_not_nan # 列ペア毎のmean
        tensor_mean2 = tensor_sum2 / n_not_nan
        """
        >>> tensor_mean1
        tensor([[0.3438, 0.6665, 0.3928, 0.0000, 0.5552],
                [0.6250, 0.5000, 0.5713,    nan, 0.0000],
                [0.3750, 1.0000, 0.4763, 0.0000, 0.5552]], device='cuda:0', dtype=torch.float16)
        >>> tensor_mean2
        tensor([[0.2856, 0.0476, 0.2856, 1.0000, 0.3809],
                [0.5000, 0.5000, 0.5000,    nan, 0.0000],
                [0.4443, 0.1666, 0.4443, 1.0000, 0.4443]], device='cuda:0', dtype=torch.float16)
        """
        # 共分散を計算する
        """
        1/n * {Sigma(xi * yi) - Sigma(xi * ym) - Sigma(yi * xm) + Sigma(xm * ym)}
        1/n * {Sigma(xi * yi) - ym*Sigma(xi) - xm*Sigma(yi) + n * xm * ym}
        """
        tensor_xiyi = torch.mm(ndf2.t(), ndf1)
        tensor_xiym = tensor_mean2 * tensor_sum1
        tensor_yixm = tensor_mean1 * tensor_sum2
        tensor_xmym = tensor_mean1 * tensor_mean2 * n_not_nan
        tensor_Sxy  = (tensor_xiyi - tensor_xiym - tensor_yixm + tensor_xmym)
        """
        >>> tensor_xiyi
        tensor([[0.2678, 0.1428, 0.5713, 0.0000, 1.0000],
                [0.2500, 0.0000, 0.1428, 0.0000, 0.0000],
                [0.4583, 0.3333, 0.7617, 0.0000, 1.0000]], device='cuda:0', dtype=torch.float16)
        >>> tensor_xiym
        tensor([[0.3928, 0.0952, 0.4487, 0.0000, 0.6343],
                [0.6250, 0.5000, 0.5713,    nan, 0.0000],
                [0.5000, 0.3333, 0.6348, 0.0000, 0.7402]], device='cuda:0', dtype=torch.float16)
        >>> tensor_yixm
        tensor([[0.3928, 0.0952, 0.4487, 0.0000, 0.6343],
                [0.6250, 0.5000, 0.5713,    nan, 0.0000],
                [0.5000, 0.3333, 0.6348, 0.0000, 0.7402]], device='cuda:0', dtype=torch.float16)
        >>> tensor_xmym
        tensor([[0.3928, 0.0952, 0.4487, 0.0000, 0.6343],
                [0.6250, 0.5000, 0.5713,    nan, 0.0000],
                [0.5000, 0.3333, 0.6348, 0.0000, 0.7402]], device='cuda:0', dtype=torch.float16)
        >>> tensor_Sxy
        tensor([[-0.1248,  0.0476,  0.1226,  0.0000,  0.3657],
                [-0.3750, -0.5000, -0.4287,     nan,  0.0000],
                [-0.0420,  0.0000,  0.1270,  0.0000,  0.2598]], device='cuda:0', dtype=torch.float16)
        """
        """
        1/n * {Sigma(xi * xi) - 2xm*Sigma(xi) + n * xm * xm}
        """
        tensor_xi2  = ndf1 * ndf1
        tensor_xi2  = torch.mm((~is_nan2).to(_dtype).t(), tensor_xi2).to(_dtype)
        tensor_Sxx  = (tensor_xi2 - 2 * (tensor_mean1 * tensor_sum1) + tensor_mean1 ** 2 * n_not_nan)
        tensor_yi2  = ndf2 * ndf2
        tensor_yi2  = torch.mm(tensor_yi2.t(), (~is_nan1).to(_dtype)).to(_dtype)
        tensor_Syy  = (tensor_yi2 - 2 * (tensor_mean2 * tensor_sum2) + tensor_mean2 ** 2 * n_not_nan)
        """
        >>> tensor_Sxx
        tensor([[0.6055, 0.6670, 0.5869, 0.0000, 0.5190],
                [0.2812, 0.5000, 0.3677,    nan, 0.0000],
                [0.5938, 0.0000, 0.5029, 0.0000, 0.5190]], device='cuda:0', dtype=torch.float16)
        >>> tensor_Syy
        tensor([[0.6943, 0.0136, 0.6943, 0.0000, 0.5854],
                [0.5000, 0.5000, 0.5000,    nan, 0.0000],
                [0.5190, 0.0555, 0.5190, 0.0000, 0.5190]], device='cuda:0', dtype=torch.float16)
        """
        # numpyに戻すと少し計算がずれる(infがあればnanにしておく)
        ndf = (tensor_Sxy / torch.sqrt(tensor_Sxx * tensor_Syy)).to(_dtype).cpu().numpy()
        ndf[ndf ==  np.inf] = np.nan
        ndf[ndf == -np.inf] = np.nan
    return ndf

