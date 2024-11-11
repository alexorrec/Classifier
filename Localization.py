import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import PRNU


def predict(prnu):
    # Return a random similarity percentage between 0 and 100
    return random.uniform(0, 100)


def prnu_localization(image_path):
    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]

    patch_size = 512

    heatmap = np.zeros((height, width), dtype=np.float32)

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = original_image[y:y + patch_size, x:x + patch_size]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue  # Skip if it's not a 512x512 patch

            prnu_patch = PRNU.extract_single(patch)

            similarity_percentage = predict(prnu_patch)

            # Normalize similarity score to range 0-1 for color mapping
            similarity_normalized = similarity_percentage / 100.0

            # Fill the corresponding location in heatmap with the normalized score
            heatmap[y:y + patch_size, x:x + patch_size] = similarity_normalized

    # Plotting with colorbar aligned to the image height
    plt.figure(figsize=(10, 10))

    # Display original image in the background
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    # Display heatmap with a color map (blue for low values, red for high)
    heatmap_plot = plt.imshow(heatmap, cmap='coolwarm', alpha=0.6, interpolation='nearest')

    # Add a colorbar that matches the height of the image
    colorbar = plt.colorbar(heatmap_plot, fraction=0.046 * height / width, pad=0.04)
    colorbar.set_label("Similarity Percentage")

    plt.axis('off')
    plt.title("Localization Heatmap (Test Mode)")
    plt.show()


# Example usage
prnu_localization('path/to/your/image.jpg')
