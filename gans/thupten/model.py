import torch
import torch.nn as nn
import torch.nn.functional as F


# based on resnet generator architecture used in the cyclegan paper
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.pad(x, (1, 1, 1, 1), mode='reflect')
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = F.pad(out, (1, 1, 1, 1), mode='reflect')
        out = self.conv2(out)
        out = self.norm2(out)
        return identity + out

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9, base_filters=64):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_filters, 7),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        in_features = base_filters
        for _ in range(2):
            out_features = in_features * 2
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ))
            in_features = out_features

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(in_features) for _ in range(num_residual_blocks)])

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for _ in range(2):
            out_features = in_features // 2
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ))
            in_features = out_features

        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, output_channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        
        # Downsampling
        for down_block in self.down_blocks:
            x = down_block(x)
            
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
            
        # Upsampling
        for up_block in self.up_blocks:
            x = up_block(x)
            
        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super().__init__()
        
        def get_discriminator_block(in_filters, out_filters, normalize=True, stride=2):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.blocks = nn.ModuleList([
            get_discriminator_block(input_channels, base_filters, normalize=False),
            get_discriminator_block(base_filters, base_filters * 2),
            get_discriminator_block(base_filters * 2, base_filters * 4),
            get_discriminator_block(base_filters * 4, base_filters * 8),
        ])
        
        self.output = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(base_filters * 8, 1, 4, padding=1)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output(x) 