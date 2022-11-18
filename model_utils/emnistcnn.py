import torch.nn as nn
import torch.nn.functional as F
import torch


class EMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 convolutional layers
        self.cv1 = nn.Conv2d(1,16,kernel_size=5, stride=1)  # input: 1 if grayscale, 3 if RGB
        self.cv2 = nn.Conv2d(16, 64, 5)
        self.cv3 = nn.Conv2d(64, 128, 5)
        self.dropout1 = nn.Dropout(0.2)
        
        # Dense layer - (fully connected)
        self.fc1 = nn.Linear(in_features=128*3*3, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=47)
        
    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        # hidden convolutional layers
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.cv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.dropout1(x)
        
        # hidden linear layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        
        return x,out

    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input
        z = self.out(z)
        out=F.log_softmax(z, dim=1)
        return z, out