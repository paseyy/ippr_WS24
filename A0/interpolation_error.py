import numpy as np
import imageio
from skimage.transform import resize
from scipy.ndimage import gaussian_filter


def mssim(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    # Standard choice for the parameters
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    truncate = 3.5
    m = 1
    C1 = (K1 * m) ** 2
    C2 = (K2 * m) ** 2

    # radius size of the local window (needed for
    # normalizing the standard deviation)
    r = int(truncate * sigma + 0.5)
    win_size = 2 * r + 1
    # use these arguments for the gaussian filtering
    # e.g. filtered = gaussian_filter(x, **filter_args)
    filter_args = {
        'sigma': sigma,
        'truncate': truncate
    }

    # Implement Eq. (9) from assignment sheet
    # S should be an "image" of the SSIM evaluated for a window 
    # centered around the corresponding pixel in the original input image
    S = np.ones_like(x)

    # crop to remove boundary artifacts, return MSSIM
    pad = (win_size - 1) // 2
    return S[pad:-pad, pad:-pad].mean()


def psnr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    # Implement Eq. (2) without for loops
    return 1.


def psnr_for(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    # Implement Eq. (2) using for loops
    return 1.


def interpolation_error():
    x = imageio.imread('./girl.png') / 255.
    shape_lower = (x.shape[0] // 2, x.shape[1] // 2)
    # downsample image to half the resolution
    # and successively upsample to the original resolution
    # using no nearest neighbor, linear and cubic interpolation
    nearest, linear, cubic = [
        resize(resize(
            x, shape_lower, order=order, anti_aliasing=False
        ), x.shape, order=order, anti_aliasing=False)
        for order in [0, 1, 3]
    ]

    for label, rescaled in zip(
        ['nearest', 'linear', 'cubic'],
        [nearest, linear, cubic]
    ):
        print(label)
        print(mssim(x, rescaled))
        print(psnr(x, rescaled))
        print(psnr_for(x, rescaled))


if __name__ == '__main__':
    interpolation_error()
