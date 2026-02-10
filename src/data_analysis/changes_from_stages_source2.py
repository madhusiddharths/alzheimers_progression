import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def compute_mean_image(stage_path):
    # List all image files in the stage directory
    image_files = [os.path.join(stage_path, img) for img in os.listdir(stage_path)]
    
    # Preallocate an array to store the sum of all images
    mean_image = np.zeros((256, 256), dtype=np.float32)  # Adjust size based on your image dimensions
    
    # Compute the sum of all images with a progress bar
    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(stage_path)}", unit="image"):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))  # Resize to a consistent size if needed
            mean_image += img.astype(np.float32)
    
    # Compute the mean by dividing by the number of images
    mean_image /= len(image_files)
    return mean_image

# Define dataset path and stages
dataset_path = "data/source_2/"
stages = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

# Compute mean images for each stage
mean_images = {}
for stage in stages:
    stage_path = os.path.join(dataset_path, stage)
    mean_images[stage] = compute_mean_image(stage_path)

# Plot mean differences between consecutive stages
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):
    # Compute the difference between consecutive stages
    diff_img = mean_images[stages[i+1]] - mean_images[stages[i]]
    
    # Plot the difference image
    axes[i].imshow(diff_img, cmap="coolwarm")
    axes[i].set_title(f"{stages[i]} → {stages[i+1]}", fontsize=12)
    axes[i].axis("off")

# Add a main title
plt.suptitle("Differences in Mean Images Between Stages", fontsize=16)
plt.tight_layout()

# plt.show()

plt.savefig("plots/source_2/changes_from_stages.png")