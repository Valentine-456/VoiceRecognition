import torch.nn as nn
import torch.nn.functional as F

from src.training.choose_activation_function import choose_activation_function

class ConvolutionBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels, 
            batch_norm_mode: str = "none",
            activation: str = "relu",
        ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if(batch_norm_mode != "none") else None
        self.batch_norm_mode = batch_norm_mode
        self.activation = choose_activation_function(activation)

    def forward(self, x):
        x = self.conv(x)

        if(self.batch_norm_mode == "none"):
            x = self.activation(x)
        elif(self.batch_norm_mode == "before"):
            x = self.bn(x)
            x = self.activation(x)
        elif(self.batch_norm_mode == "after"):
            x = self.activation(x)
            x = self.bn(x)
        else:
            raise ValueError(f"Unknown batch_norm_mode: {self.batch_norm_mode}")

        return x