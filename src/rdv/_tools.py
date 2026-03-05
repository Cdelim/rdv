import torch as _torch
from . import _core
from tqdm import tqdm

def total_variation(grid: _torch.Tensor) -> _torch.Tensor:
    """Compute the total variation of a 2D or 3D grid.

    Args:
        grid: A 2D or 3D tensor of shape (B, H, W, C) or (B, D, H, W, C).

    Returns:
        A scalar containing the total variation.
    """
    dims = grid.dim() - 2  # exclude channel dimension and batch
    if dims == 2:
        return  (grid[:, 1:, :, :] - grid[:, :-1, :, :]).abs().sum() + \
                (grid[:, :, 1:, :] - grid[:, :, :-1, :]).abs().sum()
    elif dims == 3:
        return  (grid[:, 1:, :, :, :] - grid[:, :-1, :, :, :]).abs().sum() + \
                (grid[:, :, 1:, :, :] - grid[:, :, :-1, :, :]).abs().sum() + \
                (grid[:, :, :, 1:, :] - grid[:, :, :, :-1, :]).abs().sum()
    else:
        raise ValueError("total_variation only supports 2D or 3D grids.")



from torchvision.transforms.transforms import GaussianBlur
def gaussian_filter(imgs, sigma=1.0, channel_last: bool = True, kernel_size: int = 5):
    gaussian = GaussianBlur(kernel_size, sigma=sigma)
    if channel_last:
        return gaussian(imgs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return gaussian(imgs)


def resample_3d(grids, size, mode: str = 'bilinear', align_corners=True):
    return _torch.nn.functional.interpolate(
        grids.permute(0, 4, 1, 2, 3), size=size, mode=mode, align_corners=align_corners
    ).permute(0, 2, 3, 4, 1)


def resample_2d(imgs, size, mode: str = 'bilinear', align_corners=True):
    return _torch.nn.functional.interpolate(imgs.permute(0, 3, 1, 2), size=size, mode=mode, align_corners=align_corners).permute(0, 2, 3, 1)


def accumulate(p, times, verbose=True):
    assert times > 0
    with _torch.no_grad():
        img = p()
        steps = range(1, times) if not verbose else tqdm(range(1, times), "Accumulating:")
        for i in steps:
            _torch.add(img, p(), alpha=1, out=img)
        return img/times


