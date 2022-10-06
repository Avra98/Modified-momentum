import torch
import torch.nn as nn
import torch.nn.functional as F
class smallnet(nn.Module):
    def __init__(self, channel= 3, num_classes=10, last=400, layer = 3, filter = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(last, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
