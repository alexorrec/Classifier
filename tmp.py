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


""" DEFINE SEQUENTIAL MODEL HERE 
model = tf.keras.Sequential([

    tf.keras.layers.Normalization(input_shape=shape),

    tf.keras.layers.Conv2D(8, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),

    tf.keras.layers.Conv2D(16, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),

    tf.keras.layers.Conv2D(32, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])
END SEQUENTIAL """



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
"""
DESTINATION = 'C:\\Users\\Alessandro\\Desktop\\PRNU_SET\\XL\\'
XL = 'C:\\Users\\Alessandro\\Desktop\\SD_Dataset\\XL_ds\\'

img_l = []
pp.ciclic_findings(XL, img_l)

for img in img_l:
    im = cv2.imread(img)
    min_tile = pp.get_tiles(im, just_MinMax=True)[0] # get only min tile
    residual = PRNU.extract_single(min_tile)

    cv2.imwrite(os.path.join(DESTINATION, 'PRNU_' + os.path.basename(img) +'.png'), pp.normalize(residual))
    print(f'{os.path.basename(img)} - PRNU HAS BEEN SAVED!')
"""

REALS = 'D:\\SD_Dataset\\REALS_ds'
XL = 'D:\\SD_Dataset\\XL_ds'
V2 = 'D:\\SD_Dataset\\V2_INPAINTED'
dest = 'C:\\Users\\Alessandro\\Desktop\\ELA_SET'
"""
COMPUTE AVERAGE AND SAVE DFT: REALS
"""


def main():
    for image in os.listdir(os.path.join(dest, 'naturals')):
        if '._' in image:
            os.remove(os.path.join(dest, 'naturals', image))
            print(f'removed: {image}')


if __name__ == '__main__':
    main()
