# LeNet 网络
################################
# 层名            大小      参数数目
# input           1*28*28   
# conv1           6*28*28   150+6=156  
# maxpool         6*14*14   
# conv2           16*10*10  2400+16=2416
# maxpool         16*5*5
# linear          120       48000+120=48120
# linear          84        10080+84=10164
# linear(output)  10        840+10=850
# 参数大小: 61706*4B=241KB
###################################
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import platform
import copy
import time
import torch.nn.functional as F

# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            #(in_channels, out_channels, kernel_size, stride=1, padding=0)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
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
    def forward(self, x, start_layer_idx = 0, target=0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return x, out

    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        # n_layers = 8
        # for layer_idx in range(n_layers + start_layer_idx, n_layers):
            # layer = self.layers[layer_idx]
        z = self.fc3(z)
        # if self.output_dim > 1:
        log_out = F.log_softmax(z, dim=1)
        # result = {'output': out}
        return z, log_out

class MLP(nn.Module):
    def __init__(self, class_num=10, is_complex=False):
        #继承自父类
        super(MLP, self).__init__()
        #创建一个三层的网络
        #输入的28*28为图片大小，输出的10为数字的类别数
        hidden_first = 512 if is_complex else 200
        hidden_second = 512 if is_complex else 200
        self.first = nn.Linear(in_features=28*28, out_features=hidden_first)
        self.second = nn.Linear(in_features=hidden_first, out_features=hidden_second)
        self.third = nn.Linear(in_features=hidden_second, out_features=class_num)

    def forward(self, data, start_layer_idx = 0):
        if start_layer_idx < 0: #
            return self.mapping(data, start_layer_idx=start_layer_idx)
        #先将图片数据转化为1*784的张量
        data = data.view(-1, 28*28)
        data = F.relu(self.first(data))
        data = F.relu((self.second(data)))
        # print(data.shape)
        # exit(0)
        out = self.third(data)
        # data = F.log_softmax(self.third(data), dim = 1)

        return data, out
    
    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        # n_layers = 8
        # for layer_idx in range(n_layers + start_layer_idx, n_layers):
            # layer = self.layers[layer_idx]
        z = self.third(z)
        # if self.output_dim > 1:
        out=F.log_softmax(z, dim=1)
        # result = {'output': out}
        return z, out
    
class CategoricalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.dims = [self.input_dim]
        self.dims.extend(hidden_dims)
        self.dims.append(output_dim)
        
        self.layers = nn.ModuleList([])
        
        for i in range(len(self.dims) - 1):
            ip_dim = self.dims[i]
            op_dim = self.dims[i+1]
            self.layers.append(
                nn.Linear(ip_dim, op_dim, bias=True)
            )        
            
        self.__init_net_weights__()
        
    def __init_net_weights__(self):
        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)
            
    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = x.view(-1, self.input_dim)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        out = self.layers[-1](x)
            
        return x, out

class EMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(1600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 47)

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return x, out

    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        # n_layers = 8
        # for layer_idx in range(n_layers + start_layer_idx, n_layers):
            # layer = self.layers[layer_idx]
        # z = self.fc2(z)
        z = self.fc3(z)
        # if self.output_dim > 1:
        out=F.log_softmax(z, dim=1)
        # result = {'output': out}
        return z, out

class Celeba_DNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        ### 1st hidden layer: 1920 --> 500
        self.linear_1 = nn.Linear(input_size, 500)
        ### Non-linearity in 1st hidden layer
        self.layer_1 = nn.Tanh()
        ### 2nd hidden layer: 500 --> 100
        self.linear_2 = nn.Linear(500,100)
        ### Non-linearity in 2nd hidden layer
        self.layer_2 = nn.ReLU()
        ### Output layer: 100 --> 2
        self.linear_out = nn.Linear(100, num_classes)

    def forward(self, x, start_layer_idx = 0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        ### 1st hidden layer
        out  = self.linear_1(x)
        ### Non-linearity in 1st hidden layer
        out = self.layer_1(out)
        ### 2nd hidden layer
        out  = self.linear_2(out)
        ### Non-linearity in 2nd hidden layer
        out = self.layer_2(out)
        # Linear layer (output)
        last = self.linear_out(out)
        # logits  = self.linear_out(out)
        # probas = F.softmax(logits, dim=1)
        # return logits, probas
        return out, last
    
    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        z = self.linear_out(z)
        out=F.log_softmax(z, dim=1)
        return z, out


class Celeba_CNN(nn.Module):
    def __init__(self):
        super(Celeba_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1008, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x, start_layer_idx = 0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1008)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return x, out
    
    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        z = self.fc3(z)
        out=F.log_softmax(z, dim=1)
        return z, out