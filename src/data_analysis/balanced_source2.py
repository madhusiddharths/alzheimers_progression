import os
import seaborn as sns
import matplotlib.pyplot as plt

dataset_path = "data/source_2/"
stages = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

# Count images per stage
stage_counts = {stage: len(os.listdir(os.path.join(dataset_path, stage))) for stage in stages}

# Plot distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=list(stage_counts.keys()), y=list(stage_counts.values()))
plt.title("Image Distribution Across Alzheimer's Stages")
plt.xlabel("Stages")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
# optional below
# plt.show()
if not os.path.exists("plots/source_2/"):
    os.makedirs("plots/source_2/")
plt.savefig("plots/source_2/Alzheimer_Stage_Distribution.png")