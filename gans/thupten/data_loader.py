import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import glob
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

class BrainMRIDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        image_size: Tuple[int, int] = (128, 128),
        transform: Optional[transforms.Compose] = None,
        cache_images: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # Look for multiple image formats
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            self.image_paths.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_paths.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        if not self.image_paths:
            raise ValueError(
                f"No images found in {image_dir}. "
                f"Supported formats: jpg, jpeg, png, tif, tiff"
            )
        
        self.cache_images = cache_images
        self.cached_images = {}
        
        if not self.image_paths:
            raise ValueError(f"No jpg images found in {image_dir}")
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Preload images if caching is enabled
        if self.cache_images:
            for idx, path in enumerate(self.image_paths):
                self.cached_images[idx] = Image.open(path).convert('RGB')
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache_images:
            image = self.cached_images[idx]
        else:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        
        return self.transform(image)

def get_dataloaders(
    mild_path: str,
    moderate_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (128, 128),
    cache_images: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for mild and moderate dementia images.
    
    Args:
        mild_path: Path to mild dementia images
        moderate_path: Path to moderate dementia images
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_size: Size to resize images to
        cache_images: Whether to cache images in memory
        
    Returns:
        Tuple of (mild_loader, moderate_loader)
    """
    
    # Create datasets with shared transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    mild_dataset = BrainMRIDataset(
        mild_path,
        image_size=image_size,
        transform=transform,
        cache_images=cache_images
    )
    
    moderate_dataset = BrainMRIDataset(
        moderate_path,
        image_size=image_size,
        transform=transform,
        cache_images=cache_images
    )
    
    # Create dataloaders with persistent workers
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }
    
    mild_loader = DataLoader(mild_dataset, **loader_kwargs)
    moderate_loader = DataLoader(moderate_dataset, **loader_kwargs)
    
    return mild_loader, moderate_loader 