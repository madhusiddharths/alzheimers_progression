import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import os
import time
import copy
import argparse
import matplotlib.pyplot as plt

# MLflow
import mlflow
import mlflow.pytorch

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, phase_name=""):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1} ({phase_name})')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # --- MLflow Logging ---
            metric_prefix = f"{phase_name}_" if phase_name else ""
            mlflow.log_metric(f"{metric_prefix}{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{metric_prefix}{phase}_acc", epoch_acc.item(), step=epoch)

            if phase == 'val':
                if scheduler:
                    scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'{phase_name} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    model.load_state_dict(best_model_wts)
    return model

class SafeImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        classes = [c for c in classes if not c.startswith('.')]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

class SubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self): return len(self.subset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--fast-dev-run", action="store_true", help="Run on a tiny dataset for MLflow testing")
    args = parser.parse_args()

    # MLflow Setup
    mlflow.set_experiment("Alzheimers_Classifier")

    with mlflow.start_run():
        # Log all hyperparameters
        mlflow.log_params(vars(args))
        mlflow.log_param("model_type", "EfficientNetB4")
        mlflow.log_param("img_size", 256)

        device = get_device()
        
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        full_dataset = SafeImageFolder('../data/source_2')
        class_names = full_dataset.classes

        # Limit dataset for testing
        if args.fast_dev_run:
            print("🚀 FAST DEV RUN: Using only 200 items so we can quickly generate MLflow runs!")
            subset_indices = torch.randperm(len(full_dataset))[:200]
            full_dataset = Subset(full_dataset, subset_indices)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_sub, val_sub = random_split(full_dataset, [train_size, val_size])
        
        train_dataset = SubsetWrapper(train_sub, data_transforms['train'])
        val_dataset = SubsetWrapper(val_sub, data_transforms['val'])

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        }

        # Model Building
        weights = models.EfficientNet_B4_Weights.DEFAULT
        model = models.efficientnet_b4(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        # Phase 1: Head
        for param in model.features.parameters(): param.requires_grad = False
        optimizer_head = optim.Adam(model.classifier.parameters(), lr=args.lr_head)
        model = train_model(model, dataloaders, criterion, optimizer_head, None, args.epochs_head, device, "Head")

        # Phase 2: Fine Tune
        for param in model.parameters(): param.requires_grad = True
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr_finetune)
        exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3)
        model = train_model(model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, args.epochs_finetune, device, "FineTune")

        # Save Final Model slightly differently based on run type
        local_model_path = 'efficientnet_b4_pytorch.pth'
        if not args.fast_dev_run:
            torch.save(model.state_dict(), local_model_path)
            
        print("Saving model to MLflow registry...")
        mlflow.pytorch.log_model(model, "efficientnet_classifier")

if __name__ == '__main__':
    main()
