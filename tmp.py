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
DESTINATION = '/Users/alessandrocerro/Desktop/TO_PRED_PRNU/stable-diffusion-XL'
ORIGIN = '/Users/alessandrocerro/Desktop/TO_PREDICT/AI'

img_l = []
pp.ciclic_findings(ORIGIN, img_l)

for img in img_l:
    try:
        im = cv2.imread(img)
        min_tile = pp.get_tiles(im, just_MinMax=True)[0] # get only min tile
        residual = PRNU.extract_single(min_tile)

        cv2.imwrite(os.path.join(DESTINATION, 'PRNU_' + os.path.basename(img) +'.png'), pp.normalize(residual))
        print(f'{os.path.basename(img)} - PRNU HAS BEEN SAVED!')
    except:
        pass

"""

orig = cv2.imread('/Users/alessandrocerro/PycharmProjects/CVUtilities/HIST&NOISE/Hist_Original.png')
synth = cv2.imread('/Users/alessandrocerro/PycharmProjects/CVUtilities/HIST&NOISE/Hist_PIPED.png')

plt.imsave('diff.png', synth-orig, cmap='gray')