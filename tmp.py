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

from Classifier import Tester

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
DESTINATION = '/Users/alessandrocerro/Desktop/ELA_2/naturals'
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
"""
dataset_path = '/Volumes/NO NAME/PRNU_SET'

_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            validation_split=0.2,
            subset="training",
            seed=1,
            image_size=(512,512),
            shuffle=True,
            batch_size=128
        )
val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            validation_split=0.2,
            subset="validation",
            seed=1,
            image_size=(512,512),
            shuffle=True,
            batch_size=128
        )

model = tf.keras.models.load_model('/Volumes/NO NAME/PRNU_EfficientNetB0_3v.h5')

def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(512,512))  # Load image and resize
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

TP = 0
F = 0
class_names = os.listdir(dataset_path)
for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_folder):
        for img_name in os.listdir(class_folder)[:50]:
            if '._' in img_name:
                continue
            img_path = os.path.join(class_folder, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if it's an image file
                img_array = preprocess_image(img_path)

                # Make prediction
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)  # Get the index of the class with the highest probability
                predicted_label = class_names[predicted_class]  # Map index to the class name

                if predicted_label == class_name:
                    TP += 1
                else:
                    F += 1
                # Output the prediction
                print(f'Image: {img_name} | True Class: {class_name} | Predicted Class: {predicted_label}')

print(f'TP: {TP} | F: {F}')