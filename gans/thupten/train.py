import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import os
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from pathlib import Path
import json
from typing import Dict, Any

from model import Generator, Discriminator
from data_loader import get_dataloaders

class CycleGAN:
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0
    ):
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Initialize generators and discriminators
        self.G_M2S = Generator().to(device)
        self.G_S2M = Generator().to(device)
        self.D_M = Discriminator().to(device)
        self.D_S = Discriminator().to(device)
        
        # Initialize optimizers with gradient clipping
        self.optimizer_G = optim.Adam(
            itertools.chain(self.G_M2S.parameters(), self.G_S2M.parameters()),
            lr=lr,
            betas=(beta1, beta2)
        )
        self.optimizer_D = optim.Adam(
            itertools.chain(self.D_M.parameters(), self.D_S.parameters()),
            lr=lr,
            betas=(beta1, beta2)
        )
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Initialize AMP scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, mode='min', factor=0.5, patience=20, verbose=True
        )
        self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, mode='min', factor=0.5, patience=20, verbose=True
        )

    @torch.cuda.amp.autocast()
    def train_step(self, real_mild: torch.Tensor, real_moderate: torch.Tensor) -> Dict[str, float]:
        # Move data to device
        real_mild = real_mild.to(self.device)
        real_moderate = real_moderate.to(self.device)
        
        # Train Generators
        self.optimizer_G.zero_grad(set_to_none=True)
        
        # Generate fake images
        fake_moderate = self.G_S2M(real_mild)
        fake_mild = self.G_M2S(real_moderate)
        
        # Reconstruct images
        cycle_mild = self.G_M2S(fake_moderate)
        cycle_moderate = self.G_S2M(fake_mild)
        
        # Identity mapping
        identity_mild = self.G_M2S(real_mild)
        identity_moderate = self.G_S2M(real_moderate)
        
        # Calculate generator losses
        loss_identity = (
            self.criterion_identity(identity_mild, real_mild) * self.lambda_identity +
            self.criterion_identity(identity_moderate, real_moderate) * self.lambda_identity
        )
        
        loss_GAN = (
            self.criterion_GAN(self.D_M(fake_moderate), torch.ones_like(self.D_M(fake_moderate))) +
            self.criterion_GAN(self.D_S(fake_mild), torch.ones_like(self.D_S(fake_mild)))
        ) * 0.5
        
        loss_cycle = (
            self.criterion_cycle(cycle_mild, real_mild) * self.lambda_cycle +
            self.criterion_cycle(cycle_moderate, real_moderate) * self.lambda_cycle
        )
        
        loss_G = loss_identity + loss_GAN + loss_cycle
        
        # Backward pass with AMP
        self.scaler.scale(loss_G).backward()
        self.scaler.step(self.optimizer_G)
        
        # Train Discriminators
        self.optimizer_D.zero_grad(set_to_none=True)
        
        # Discriminator losses
        loss_D_mild = (
            self.criterion_GAN(self.D_S(real_mild), torch.ones_like(self.D_S(real_mild))) +
            self.criterion_GAN(self.D_S(fake_mild.detach()), torch.zeros_like(self.D_S(fake_mild)))
        ) * 0.5
        
        loss_D_moderate = (
            self.criterion_GAN(self.D_M(real_moderate), torch.ones_like(self.D_M(real_moderate))) +
            self.criterion_GAN(self.D_M(fake_moderate.detach()), torch.zeros_like(self.D_M(fake_moderate)))
        ) * 0.5
        
        loss_D = loss_D_mild + loss_D_moderate
        
        # Backward pass with AMP
        self.scaler.scale(loss_D).backward()
        self.scaler.step(self.optimizer_D)
        
        # Update scaler
        self.scaler.update()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_identity': loss_identity.item(),
            'loss_GAN': loss_GAN.item()
        }

    def save_checkpoint(
        self,
        epoch: int,
        losses: Dict[str, float],
        save_path: str = 'checkpoints'
    ) -> None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'G_M2S_state_dict': self.G_M2S.state_dict(),
            'G_S2M_state_dict': self.G_S2M.state_dict(),
            'D_M_state_dict': self.D_M.state_dict(),
            'D_S_state_dict': self.D_S.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'losses': losses
        }
        
        torch.save(checkpoint, save_path / f'cyclegan_epoch_{epoch}.pth')
        
        # Save latest checkpoint separately
        torch.save(checkpoint, save_path / 'latest.pth')

def main():
    # Get absolute paths for data directories
    current_dir = Path(__file__).parent.absolute()
    mild_path = (current_dir / '../../data/source_2/Mild Dementia').resolve()
    moderate_path = (current_dir / '../../data/source_2/Moderate Dementia').resolve()
    
    print(f"Looking for images in:\n{mild_path}\n{moderate_path}")
    
    if not mild_path.exists():
        raise ValueError(f"Directory not found: {mild_path}")
    if not moderate_path.exists():
        raise ValueError(f"Directory not found: {moderate_path}")
    
    # Training configuration
    config = {
        'mild_path': str(mild_path),
        'moderate_path': str(moderate_path),
        'batch_size': 1,
        'num_epochs': 200,
        'save_interval': 10,
        'image_size': (128, 128),
        'num_workers': 4,
        'lr': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'lambda_cycle': 10.0,
        'lambda_identity': 5.0
    }
    
    print(f"Starting training with config: {config}")
    
    # Create samples directory in thupten folder
    samples_dir = current_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    # Get dataloaders
    mild_loader, moderate_loader = get_dataloaders(
        config['mild_path'],
        config['moderate_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    # Initialize model
    model = CycleGAN(
        lr=config['lr'],
        beta1=config['beta1'],
        beta2=config['beta2'],
        lambda_cycle=config['lambda_cycle'],
        lambda_identity=config['lambda_identity']
    )
    
    for epoch in range(config['num_epochs']):
        progress_bar = tqdm(
            enumerate(zip(mild_loader, moderate_loader)),
            total=min(len(mild_loader), len(moderate_loader)),
            desc=f"Epoch {epoch}/{config['num_epochs']}"
        )
        
        epoch_losses = []
        for i, (mild_batch, moderate_batch) in progress_bar:
            losses = model.train_step(mild_batch, moderate_batch)
            epoch_losses.append(losses)
            
            if i % 100 == 0:
                with torch.no_grad():
                    fake_moderate = model.G_S2M(mild_batch.to(model.device))
                    save_image(
                        torch.cat([mild_batch, fake_moderate.cpu()], dim=0),
                        samples_dir / f'epoch_{epoch}_batch_{i}.png',
                        normalize=True
                    )
            
            progress_bar.set_description(
                f"Epoch {epoch}/{config['num_epochs']} "
                f"Loss_G: {losses['loss_G']:.4f} "
                f"Loss_D: {losses['loss_D']:.4f}"
            )
        
        # Calculate average losses for the epoch
        avg_losses = {
            k: np.mean([loss[k] for loss in epoch_losses])
            for k in epoch_losses[0].keys()
        }
        
        # Update learning rate schedulers
        model.scheduler_G.step(avg_losses['loss_G'])
        model.scheduler_D.step(avg_losses['loss_D'])
        
        print(f"Epoch {epoch} - Losses: {avg_losses}")

if __name__ == '__main__':
    main() 