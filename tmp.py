import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PreProcessing as pp
# import cv2
import tensorflow as tf
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

import numpy as np
import cv2

folders = ['C:\\Users\\Alessandro\\Desktop\\PRNU_NPY\\naturals',
           'C:\\Users\\Alessandro\\Desktop\\PRNU_NPY\\stable-diffusion']
dest = ['C:\\Users\\Alessandro\\Desktop\\PRNU_BALANCED\\naturals',
        'C:\\Users\\Alessandro\\Desktop\\PRNU_BALANCED\\stable-diffusion']

for folder in folders:
    for img_p in os.listdir(folder):
        grayscale_image = np.load(os.path.join(folder, img_p))  # Replace with your actual file path
        print('processing', os.path.basename(img_p)[:-4])

        # Step 2: Normalize the image values to range [0, 255]
        # First, shift the image values to be non-negative by adding the absolute value of the minimum
        min_val = np.min(grayscale_image)
        max_val = np.max(grayscale_image)

        # Normalize the image to the range [0, 1]
        normalized_image = (grayscale_image - min_val) / (max_val - min_val)

        # Scale it to the range [0, 255]
        normalized_image = normalized_image * 255

        # Step 3: Convert the image to uint8 (8-bit unsigned integer)
        normalized_image = normalized_image.astype(np.uint8)

        # Step 4: Save the normalized image as a .png file
        cv2.imwrite(os.path.join(dest[folders.index(folder)], os.path.basename(img_p)[:-4]+'.png'),
                    normalized_image)
