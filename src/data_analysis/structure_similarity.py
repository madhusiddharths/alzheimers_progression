from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def compute_ssim_between_stages(stage1_path, stage2_path):
    stage1_images = random.sample(os.listdir(stage1_path), 5)
    stage2_images = random.sample(os.listdir(stage2_path), 5)
    
    ssim_scores = []
    
    for img1, img2 in zip(stage1_images, stage2_images):
        img1_path = os.path.join(stage1_path, img1)
        img2_path = os.path.join(stage2_path, img2)

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        score, _ = ssim(img1, img2, full=True)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)
dataset_path = "/home/jchmura8/workspace/github.com/johnchmura/CS584/CS584-Project/data/source 2/"
stages = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

for i in range(3):
    stage1, stage2 = stages[i], stages[i+1]
    ssim_score = compute_ssim_between_stages(os.path.join(dataset_path, stage1), os.path.join(dataset_path, stage2))
    print(f"SSIM between {stage1} and {stage2}: {ssim_score:.4f}")
