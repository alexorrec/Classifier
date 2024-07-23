import cv2
from PIL import Image, ImageEnhance, ImageChops
import os
import numpy as np


def img2gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def dft(image):
    img = img2gray(image)
    _dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(_dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum


def get_tiles(img: np.ndarray, size=(512, 512)):
    height, width, channels = img.shape

    crop_height, crop_width = size
    weights: list = list()

    for y in range(0, height - crop_height + 1, 512):
        for x in range(0, width - crop_width + 1, 512):
            crop = img[y:y + crop_height, x:x + crop_width]
            crop_sum = np.sum(dft(crop))
            weights.append(((y, x), crop_sum))

    min_crop = min(weights, key=lambda l: l[1])
    max_crop = max(weights, key=lambda l: l[1])

    y_min, x_min = min_crop[0]
    y_max, x_max = max_crop[0]
    min_tile = img[y_min:y_min + crop_height, x_min:x_min + crop_width]
    max_tile = img[y_max:y_max + crop_height, x_max:x_max + crop_width]

    return max_tile, min_tile


def get_ELA_(cv2_image, save_dir: str = '', offset: int = 10, brigh_it_up: int = 1):
    im = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    tmp_fname = os.path.join(save_dir, 'TMP_EXT')
    im.save(tmp_fname, 'JPEG', quality=95)
    tmp_fname_im = Image.open(tmp_fname)

    ela_im = ImageChops.difference(im, tmp_fname_im)

    if brigh_it_up > 1:
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / (max_diff / brigh_it_up)
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    ela_cv2_image = cv2.cvtColor(np.array(ela_im), cv2.COLOR_RGB2BGR)
    os.remove(tmp_fname)

    return ela_cv2_image
