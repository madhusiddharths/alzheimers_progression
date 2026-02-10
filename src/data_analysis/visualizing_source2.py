import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def show_sample_images(dataset_path, stages, num_samples=5):
    fig, axes = plt.subplots(len(stages), num_samples, figsize=(15, 10))
    
    for i, stage in enumerate(stages):
        stage_path = os.path.join(dataset_path, stage)
        image_files = random.sample(os.listdir(stage_path), num_samples)

        for j, img_file in enumerate(image_files):
            img_path = os.path.join(stage_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[i, j].imshow(img)
            axes[i, j].axis("off")
        
        # Add stage label to the first column of each row
        axes[i, 0].text(-50, 50, stage, rotation=0, va="center", ha="right", fontsize=12, color="black")
    
    plt.suptitle("Sample Images from Each Stage", fontsize=16)
    plt.tight_layout()
    plt.show()

dataset_path = "/home/jchmura8/workspace/github.com/johnchmura/CS584/CS584-Project/data/source 2/"
stages = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

show_sample_images(dataset_path, stages)