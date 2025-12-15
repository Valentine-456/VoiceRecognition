import torch.nn as nn
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels, batch_norm_mode: str = "none"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if(batch_norm_mode != "none") else None
        self.batch_norm_mode = batch_norm_mode

    def forward(self, x):
        if(self.batch_norm_mode == "none"):
            x = self.conv(x)
            x = F.relu(x)
        elif(self.batch_norm_mode == "before"):
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)
        elif(self.batch_norm_mode == "after"):
            x = self.conv(x)
            x = F.relu(x)
            x = self.bn(x)
        else:
            raise ValueError(f"Unknown batch_norm_mode: {self.batch_norm_mode}")

        return x