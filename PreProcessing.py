import random
import time

import cv2
from PIL import Image, ImageEnhance, ImageChops
import os
import numpy as np


def ciclic_findings(path, listed_):
    """Given a Path, find all images even in subpaths"""
    for _ in os.listdir(path):
        if os.path.isdir(os.path.join(path, _)):
            ciclic_findings(os.path.join(path, _), listed_)
        elif os.path.isfile(os.path.join(path, _)) and 'mask' not in _ and '._' not in _ and '.DS'not in _:
            listed_.append(os.path.join(path, _))
            #print(f'PATH: {os.path.join(path, _)}')

def enhance_peaks(img, kernel_):
    kern = np.ones((kernel_, kernel_), np.uint8)
    return cv2.dilate(img, kern)

def img2gray(image):
    """RGB to GRAYSCALE"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize(res):
    normalized_image = (res - np.min(res)) * (255.0 / (np.max(res) - np.min(res)))
    return normalized_image.astype('uint8')


def dft(image):
    """PERFORM LOGGED DFT"""
    img = img2gray(image)
    _dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(_dft)
    magnitude_spectrum = 15 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum


def get_tiles(img: np.ndarray, size=(512, 512), more_tiles=0, just_MinMax=False):
    """Get a Crops of the original image"""

    height, width, channels = img.shape

    crop_height, crop_width = size
    weights: list = list()

    for y in range(0, height - crop_height + 1, 512):
        for x in range(0, width - crop_width + 1, 512):
            crop = img[y:y + crop_height, x:x + crop_width]
            crop_sum = np.sum(dft(crop))
            weights.append(((y, x), crop_sum))

    tiles = []
    if just_MinMax:
        min_crop = min(weights, key=lambda l: l[1])
        max_crop = max(weights, key=lambda l: l[1])

        y_min, x_min = min_crop[0]
        y_max, x_max = max_crop[0]

        min_tile = img[y_min:y_min + crop_height, x_min:x_min + crop_width]
        max_tile = img[y_max:y_max + crop_height, x_max:x_max + crop_width]

        #weights.remove(min_crop)
        #weights.remove(max_crop)

        tiles.append(min_tile)
        tiles.append(max_tile)
        return tiles

    random.shuffle(weights)

    while len(tiles) < more_tiles:
        tile_crop = random.choice(weights)
        weights.remove(tile_crop)
        y_min, x_min = tile_crop[0]
        tile = img[y_min:y_min + crop_height, x_min:x_min + crop_width]
        tiles.append(tile)

    return tiles


def get_ELA_(cv2_image, save_dir: str = '', brigh_it_up: int = 1):
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


def export_jpg(img: np.array, filenname: str, quality: int = 95, path: str = ''):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im.save(os.path.join(path, filenname) + '.jpg', format='JPEG', quality=quality)
    print(f'{filenname} SAVED!')


def average_images(img_l: list):
    """get a list of cv2, return a cv2"""
    N = len(img_l)

    arr = img_l[0].astype(np.float32)
    for im in img_l[1:]:
        arr += im.astype(np.float32)

        print(f'\r Computing: {int(time.time_ns() + 1) * 100 // N}%', end='')
        time.sleep(0.01)

    arr = arr / N

    return arr.astype(np.uint8)


def get_NoisePrint(img: np.array):
    pass


def get_SensorPatternNoise(img: np.array):
    pass


def get_PRNU(img: np.array):
    pass
