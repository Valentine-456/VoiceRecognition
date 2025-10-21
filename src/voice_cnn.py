import torch.nn as nn
import torch.nn.functional as F


class VoiceCNN(nn.Module):
    def __init__(self):
        super(VoiceCNN,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #ajust for spectrogram size
        self.fc2 = nn.Linear(120, 2)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
       
        x = self.fc2(x)
        return x


net = Net()
