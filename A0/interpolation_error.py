import numpy as np
import imageio
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
#import matplotlib.pyplot as plt

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
    x_pad = np.pad(x, ((r, r), (r, r)), "edge")
    y_pad = np.pad(y, ((r, r), (r, r)), "edge")

    r_squared = r ** 2
    n_wave = r_squared / (r_squared - 1)

    S = np.ones_like(x)
    for row in range(S.shape[0]):
        for col in range(S.shape[1]):
            patch_x = x_pad[row:row+2*r, col:col+2*r]
            patch_y = y_pad[row:row+2*r, col:col+2*r]

            mu_x = gaussian_filter(patch_x, **filter_args).mean()
            mu_y = gaussian_filter(patch_y, **filter_args).mean()

            mu_x_squared = mu_x ** 2
            mu_y_squared = mu_y ** 2

            sigma_x_squared = n_wave * gaussian_filter((patch_x - mu_x) ** 2, **filter_args).sum()
            sigma_y_squared = n_wave * gaussian_filter((patch_y - mu_y) ** 2, **filter_args).sum()

            sigma_xy = n_wave * gaussian_filter((patch_x - mu_x) * (patch_y - mu_y), **filter_args).sum()

            S[row, col] = (((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) /
                           ((mu_x_squared + mu_y_squared + C1) * (sigma_x_squared + sigma_y_squared + C2)))

    # crop to remove boundary artifacts, return MSSIM
    pad = (win_size - 1) // 2
    return S[pad:-pad, pad:-pad].mean()



def psnr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    # Implement Eq. (2) without for loops
    mse = np.mean(np.square(x - y))

    return 10 * np.log10(1 / mse)


def psnr_for(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    # Implement Eq. (2) using for loops
    num_rows = x.shape[0]
    num_cols = x.shape[1]
    ase = 0

    for row in range(num_rows):
        for col in range(num_cols):
            diff = x[row, col] - y[row, col]
            ase += diff * diff

    mse = ase / (num_rows * num_cols)

    return 10 * np.log10(1 / mse)


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

    #_, ax = plt.subplots(1, 3, figsize=(9, 3))
    #ax[0].imshow(nearest, cmap='gray')
    #ax[1].imshow(linear, cmap='gray')
    #ax[2].imshow(cubic, cmap='gray')
    #plt.show()


if __name__ == '__main__':
    interpolation_error()
