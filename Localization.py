from tkinter import Image
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PRNU
import tensorflow as tf
import os
import PreProcessing as pp
from tqdm import tqdm

model_path = '/Volumes/NO NAME/PRNULOC_1001.keras'
model = tf.keras.models.load_model(model_path)


def get_ground(path):
    if path[-4:] == '.png':
        return path[:-4] + 'mask.png'
    return None


def predict(_patch):
    img_array = tf.keras.utils.img_to_array(_patch)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = np.repeat(img_array, repeats=3, axis=-1)
    pred = model.predict(img_array, verbose=0)
    if pred[0][0] >= 0.8:
        return 90
    elif 0.65 <= pred[0][0] < 0.8:
        return 60
    elif 0.5 < pred[0][0] < 0.65:
        return 35
    elif 0.35 < pred[0][0] <= 0.5:
        return 5
    return -10


def prnu_localization(image_path, mask_path):
    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]

    patch_size = 256
    stride = 32
    heatmap = np.zeros((height, width), dtype=np.float32)
    i = 0
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            patch = original_image[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            prnu_patch = pp.normalize(PRNU.extract_single(patch))
            score = predict(prnu_patch)
            heatmap[y:y + patch_size, x:x + patch_size] += score
            if np.any(heatmap[y:y + patch_size, x:x + patch_size] < 0):
                heatmap[y:y + patch_size, x:x + patch_size] = 0

            """            
            plt.figure(figsize=(12, 10))
            #plt.subplot(1, 1, 1)  # 1 rows, 2 column, 1st subplot
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            heatmap_plot = plt.imshow(heatmap, alpha=0.5)
            colorbar = plt.colorbar(heatmap_plot, fraction=0.046 * height / width, pad=0.04)
            colorbar.set_label("Denoised to Inpainted")
            plt.axis('off')
            i += 1
            plt.savefig(os.path.join(os.path.basename(img_p), f'{str(i)}_PLOT.jpg'))
            plt.cla()
            plt.clf()
            plt.close('all')
            """
            cv2.imwrite('test/test.png', heatmap)


folder = '/Users/alessandrocerro/Desktop/presentation'
imgs_ = []
pp.ciclic_findings(folder, imgs_)
for img_p in tqdm(imgs_, desc=f'Processing folder: '):
    mask = get_ground(img_p)
    if mask is not None:
        os.mkdir(os.path.join('', os.path.basename(img_p)))
        print(img_p, mask)
        prnu_localization(img_p, mask)

##########################################
"""
path = '/Users/alessandrocerro/Desktop/TO_PRED_PRNU/stable-diffusion-XL'
for image_p in os.listdir(path)[:100]:
    _patch = cv2.imread(os.path.join(path, image_p))
    label, score = predict(_patch)
    print(f'{image_p} - label: {label} - score: {score}')
"""
