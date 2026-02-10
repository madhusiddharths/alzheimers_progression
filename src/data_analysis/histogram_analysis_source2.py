import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def plot_intensity_histogram(stage_path, stage_name, ax):
    img_files = random.sample(os.listdir(stage_path), 5)
    
    for img_file in img_files:
        img_path = os.path.join(stage_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ax.hist(img.ravel(), bins=256, alpha=0.5, label=img_file)

    ax.set_title(f"Pixel Intensity Distribution - {stage_name}")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_xlim(0, 160)
    ax.set_ylim(0,5000)


# Define dataset path and stages
dataset_path = "/home/jchmura8/workspace/github.com/johnchmura/CS584/CS584-Project/data/source 2/"
stages = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows, 2 columns for 4 stages
axes = axes.ravel()  # Flatten the axes array for easy iteration

# Plot histograms for each stage
for i, stage in enumerate(stages):
    plot_intensity_histogram(os.path.join(dataset_path, stage), stage, axes[i])

# Adjust layout and display the figure
plt.tight_layout()
plt.show()