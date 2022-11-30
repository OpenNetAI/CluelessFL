import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            #(in_channels, out_channels, kernel_size, stride=1, padding=0)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2)   #output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(84, 10))

    # 定义前向传播过程，输入为x
    def forward(self, x, start_layer_idx = 0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return x, out

    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input 
        z = self.fc3(z)
        out=F.log_softmax(z, dim=1)
        return z,out
