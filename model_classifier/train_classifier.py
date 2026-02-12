import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import time
import copy
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = '../data/source_2'
IMG_SIZE = (256, 256) # EfficientNetB4 default is 380, but 256 is fine for speed
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 10
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
MODEL_SAVE_PATH = 'efficientnet_b4_pytorch.pth'

def get_device():
    if torch.backends.mps.is_available():
        print("✅ MPS (Mac GPU) Detected. Using Metal Performance Shaders.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ CUDA GPU Detected.")
        return torch.device("cuda")
    else:
        print("⚠️ No GPU detected. Running on CPU (will be slower).")
        return torch.device("cpu")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu', save_path='best_model.pth'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler:
                # Note: ReduceLROnPlateau step should be called after validation loss
                pass

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())
                if scheduler:
                     scheduler.step(epoch_loss)

                # Deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
                    print(f"  New best model saved with Acc: {best_acc:.4f}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

# Create a wrapper dataset to ignore hidden files/folders
class SafeImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        # Filter out hidden directories (starting with .)
        classes = [c for c in classes if not c.startswith('.')]
        # Rebuild class_to_idx
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

# Wrapper to apply transforms to subsets
class SubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def main():
    device = get_device()
    
    # 1. Data Augmentation and Normalization
    # EfficientNet expects normalization, usually standard ImageNet mean/std
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), # Slight rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Light color aug
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print(f"Searching for data in: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found.")
        return

    full_dataset = SafeImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")

    # Split: 80% train, 20% val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # We reload ImageFolder without transforms first to avoid double transform
    # Actually SafeImageFolder already does it without transform if we don't pass it.
    
    # Random split indices
    train_sub, val_sub = random_split(full_dataset, [train_size, val_size])
    
    train_dataset = SubsetWrapper(train_sub, data_transforms['train'])
    val_dataset = SubsetWrapper(val_sub, data_transforms['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }

    # 2. Model Setup (EfficientNet B4)
    print("\n--- Building EfficientNetB4 Model ---")
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)

    # Modify Classifier Head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # --- PHASE 1: Train Head Only ---
    print("\n=== PHASE 1: Training Head (Freezing features) ===")
    
    # Freeze all layers except classifier
    for param in model.features.parameters():
        param.requires_grad = False
    
    optimizer_head = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE_HEAD)
    
    model, t_acc, v_acc, t_loss, v_loss = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer_head, 
        scheduler=None, # No scheduler for short head training
        num_epochs=EPOCHS_HEAD, 
        device=device,
        save_path=MODEL_SAVE_PATH
    )

    # --- PHASE 2: Fine-Tuning ---
    print("\n=== PHASE 2: Fine-Tuning (Unfreezing all) ===")
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_ft = optim.Adam(model.parameters(), lr=LEARNING_RATE_FINE_TUNE)
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode='min', factor=0.1, patience=3
    )

    model, t_acc_ft, v_acc_ft, t_loss_ft, v_loss_ft = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer_ft, 
        scheduler=exp_lr_scheduler, 
        num_epochs=EPOCHS_FINE_TUNE, 
        device=device,
        save_path=MODEL_SAVE_PATH
    )
    
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    # Plotting
    try:
        full_t_acc = t_acc + t_acc_ft
        full_v_acc = v_acc + v_acc_ft
        full_t_loss = t_loss + t_loss_ft
        full_v_loss = v_loss + v_loss_ft

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(full_t_acc, label='Training Accuracy')
        plt.plot(full_v_acc, label='Validation Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(full_t_loss, label='Training Loss')
        plt.plot(full_v_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig('training_history_pytorch.png')
        print("Saved training graph to training_history_pytorch.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == '__main__':
    main()
