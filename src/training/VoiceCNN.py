import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.choose_activation_function import choose_activation_function
from src.training.ConvolutionBlock import ConvolutionBlock


class VoiceCNN(nn.Module):
    def __init__(
            self, 
            dropout_rate: int = 0.3, 
            in_channels: int = 3, 
            num_classes: int = 2, 
            batch_norm_mode: str = "none",
            activation: str = "relu",
        ):
        super().__init__()
        self.conv1 = ConvolutionBlock(in_channels, 16, batch_norm_mode, activation)
        self.conv2 = ConvolutionBlock(16, 32, batch_norm_mode, activation)
        self.conv3 = ConvolutionBlock(32, 64, batch_norm_mode, activation)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            choose_activation_function(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.conv3(x)
        x = self.dropout(x)
        return self.head(x)
