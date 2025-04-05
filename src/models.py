import torch
import torch.nn as nn
import math

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class ESPCN(nn.Module):
    def __init__(self, scale_factor=2):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(32, scale_factor**2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(self.conv3(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

class EDSR(nn.Module):
    def __init__(self, scale_factor=2, num_blocks=8):
        super(EDSR, self).__init__()
        
        # Initial convolution
        self.conv_first = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )
        
        # Upscaling
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(64 * (scale_factor ** 2)),
            nn.PixelShuffle(scale_factor)
        )
        
        # Final convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.residual_blocks(x)
        x = self.upscale(x)
        x = self.conv_last(x)
        return x

# Channel Attention Module for RCAN
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Residual Channel Attention Block
class RCAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelAttention(channels, reduction)
        )

    def forward(self, x):
        return x + self.body(x)

# Residual Group
class ResidualGroup(nn.Module):
    def __init__(self, channels, num_blocks, reduction=16):
        super(ResidualGroup, self).__init__()
        modules = [RCAB(channels, reduction) for _ in range(num_blocks)]
        modules.append(nn.Conv2d(channels, channels, 3, padding=1))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.body(x)

# Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, scale_factor=2, num_groups=10, num_blocks=20, channels=64, reduction=16):
        super(RCAN, self).__init__()
        
        # Initial convolution
        self.conv_first = nn.Conv2d(1, channels, 3, padding=1)
        
        # Residual Groups
        self.residual_groups = nn.Sequential(
            *[ResidualGroup(channels, num_blocks, reduction) for _ in range(num_groups)]
        )
        self.conv_mid = nn.Conv2d(channels, channels, 3, padding=1)
        
        # Upsampling
        modules_tail = [
            nn.Conv2d(channels, channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(channels, 1, 3, padding=1)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.conv_first(x)
        res = x
        x = self.residual_groups(x)
        x = self.conv_mid(x)
        x = x + res
        x = self.tail(x)
        return x

# SRResNet
class SRResNet(nn.Module):
    def __init__(self, scale_factor=2, num_blocks=16):
        super(SRResNet, self).__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        resnet_blocks = []
        for _ in range(num_blocks):
            resnet_blocks.append(ResidualBlock(64))
        self.residual_blocks = nn.Sequential(*resnet_blocks)
        
        # Post-residual convolution
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        
        # Upsampling
        upsampling = []
        for _ in range(int(math.log2(scale_factor))):
            upsampling.append(nn.Conv2d(64, 256, kernel_size=3, padding=1))
            upsampling.append(nn.PixelShuffle(2))
            upsampling.append(nn.ReLU(inplace=True))
        self.upsampling = nn.Sequential(*upsampling)
        
        # Output convolution
        self.conv_output = nn.Conv2d(64, 1, kernel_size=9, padding=4)
        
    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual_blocks(out)
        out = self.bn_mid(self.conv_mid(out))
        out = out + residual
        out = self.upsampling(out)
        out = self.conv_output(out)
        return out
