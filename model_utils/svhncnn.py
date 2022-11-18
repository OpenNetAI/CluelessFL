import torch.nn as nn
import torch.nn.functional as F


class SVHNCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(2304, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return x,out

    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        z = self.fc3(z)
        out=F.log_softmax(z, dim=1)
        return z, out