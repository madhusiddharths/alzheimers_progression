import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, random_split
import os
import sys

DATA_DIR = '../data/source_2'
IMG_SIZE = (256, 256)
BATCH_SIZE = 64
MODEL_PATH = 'efficientnet_b4_pytorch.pth'
NUM_TEST_IMAGES = 2048 # evaluate on 2048 images for speed

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class SafeImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        classes = [c for c in classes if not c.startswith('.')]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    device = get_device()
    print(f"Using device: {device}")
    
    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = SafeImageFolder(DATA_DIR, transform=val_transforms)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    
    # Take a random subset for faster evaluation
    if len(full_dataset) > NUM_TEST_IMAGES:
        generator = torch.Generator().manual_seed(42)
        dataset, _ = random_split(full_dataset, [NUM_TEST_IMAGES, len(full_dataset) - NUM_TEST_IMAGES], generator=generator)
    else:
        dataset = full_dataset
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    running_corrects = 0
    total = 0
    print(f"Evaluating on {len(dataset)} random images...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            if i % 5 == 0:
                print(f"Batch {i}/{len(dataloader)}")
            
    accuracy = running_corrects.float() / total
    print(f"\nFinal Estimated Accuracy (on {total} images): {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
