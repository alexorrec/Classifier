import numpy as np
import cv2
import matplotlib.pyplot as plt
import PRNU
import tensorflow as tf
import os
import PreProcessing as pp

model_path = '/Volumes/NO NAME/LAST_PRNU_EffNet0_2v.h5'
model = tf.keras.models.load_model(model_path)

# Load pre-stored Labels
with open('labels2v.txt', 'r') as file:
    labels = [line.strip() for line in file]

print(f'Loaded model: {os.path.basename(model_path)} - model Labels: {labels}')


def predict(_patch):
    img_array = tf.keras.utils.img_to_array(_patch)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    pred = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(pred[0])
    label = labels[np.argmax(score)]
    return label, 100 * np.max(score)


def most_frequent(List):
    return max(set(List), key=List.count)


def prnu_localization(image_path):
    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]

    patch_size = 512

    heatmap = np.zeros((height, width), dtype=np.float32)
    patch_labels: list = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = original_image[y:y + patch_size, x:x + patch_size]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue  # Not a 512 patch

            print(f'processing patch @ {x, y}')
            prnu_patch = PRNU.extract_single(patch)

            label, score = predict(prnu_patch)

            norm_score = score / 100.0
            patch_labels.append(label)
            if label == 'naturals':
                norm_score = (100 - norm_score) / 100.0

            heatmap[y:y + patch_size, x:x + patch_size] = norm_score

    plt.figure(figsize=(10, 8))

    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    heatmap_plot = plt.imshow(heatmap, cmap='coolwarm', alpha=0.6, interpolation='nearest')

    colorbar = plt.colorbar(heatmap_plot, fraction=0.046 * height / width, pad=0.04)
    colorbar.set_label("nat 2 AI")

    plt.axis('off')
    plt.title(f"Localization Heatmap")
    plt.show()

prnu_localization('/Users/alessandrocerro/Desktop/TO_PREDICT/AI/D24_L6S2C2.JPG/D24_L6S2C2.JPG_0.png')

##########################################
"""
path = '/Users/alessandrocerro/Desktop/TO_PRED_PRNU/stable-diffusion-XL'
for image_p in os.listdir(path)[:100]:
    _patch = cv2.imread(os.path.join(path, image_p))
    label, score = predict(_patch)
    print(f'{image_p} - label: {label} - score: {score}')
"""
