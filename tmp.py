import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PreProcessing as pp
# import cv2
# import tensorflow as tf
import random
import PRNU
from PIL import Image

"""
ORIGIN = 'D:\\TO_PRED_MIN\\REALS'
destination = 'C:\\Users\\Alessandro\\Desktop\\TO_PRED_JPG\\REALS'

for path in os.listdir(ORIGIN):
    im = cv2.imread(os.path.join(ORIGIN, path))
    pp.export_jpg(im, os.path.basename(path), path=destination)
"""

"""DESTINATION = 'C:\\Users\\Alessandro\\Desktop\\TO_PRED_JPG\\AI_XL\\'
REALS = 'C:\\Users\\Alessandro\\Desktop\\JPG_T'

for img in os.listdir(REALS):
    im = cv2.imread(os.path.join(REALS, img))
    max_tile, min_tile = pp.get_tiles(im)
    cv2.imwrite(DESTINATION + 'min_' + img, pp.get_ELA_(min_tile))
    cv2.imwrite(DESTINATION + 'max_' + img, pp.get_ELA_(max_tile))"""


def normalize(res):
    normalized_image = (res - np.min(res)) * (255.0 / (np.max(res) - np.min(res)))
    return normalized_image.astype('uint8')


path = '/Users/alessandrocerro/Desktop/D05_L1S4C3_0.jpg'
im = np.asarray(Image.open(path))
tiles = pp.get_tiles(im, just_MinMax=True)

for img in tiles:
    residual = PRNU.extract_single(img)
    fname = str(random.randint(0, 10))
    cv2.imwrite('cv2_' + fname + '.png', normalize(residual))
