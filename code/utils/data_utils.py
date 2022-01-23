import random
import torch
import numpy as np
import math
import cv2


def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    ix = random.randrange(0, iw - ip + 1) if iw > ip else 0
    iy = random.randrange(0, ih - ip + 1) if ih > ip else 0
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        img = img.astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)

        return tensor

    return [_np2Tensor(a) for a in args]


def data_augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img)

        return img

    return [_augment(a) for a in args]


def bicubic_resize(img, size=None):
    h, w, c = img.shape
    if size is None:
        size = (w, h)
    bic_img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return bic_img


def postprocess(*images, rgb_range):
    def _postprocess(img, rgb_coefficient):
        img = img.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data.cpu().numpy()  # BCHW -> CHW
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)  # CHW -> HWC
        if img.shape[2] == 1:
            img = np.concatenate((img, img, img), axis=2)

        return img

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient) for img in images]


def calc_psnr(img1, img2, rgb_range=1., shave=4):
    if isinstance(img1, torch.Tensor):
        img1 = img1[:, :, shave:-shave, shave:-shave]
        img1 = img1.to('cpu').numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2[:, :, shave:-shave, shave:-shave]
        img2 = img2.to('cpu').numpy()
    mse = np.mean((img1 / rgb_range - img2 / rgb_range) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
