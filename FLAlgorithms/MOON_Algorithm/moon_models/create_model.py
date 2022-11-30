from .models import LeNet, MLP, EMNISTCNN, Celeba_DNN, Celeba_CNN, CategoricalMLP
from .cifar_model import ResNet18, AlexNet_CIFAR10, CNNNet
import torch.nn as nn
import math
NUMBER_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 64
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
DROPOUT = 0.3

def create_mlp_model(model, input_dim, hidden_dims, output_dim, num_classes=10):
    created_model = CategoricalMLP(input_dim, hidden_dims, output_dim).cuda()
    return created_model

def create_model(model, num_classes=10):
    created_model = {
        "lenet": LeNet().cuda(),
        "resnet18": ResNet18().cuda(),
        "alexnet": AlexNet_CIFAR10().cuda(),
        "mlp": MLP(class_num=num_classes).cuda(),
        "cnnnet": CNNNet(), 
        "emnistcnn": EMNISTCNN(),
        "celeba_dnn": Celeba_DNN(input_size = 48*40, num_classes = 2).cuda(),
        "celeba_cnn": Celeba_CNN()
    }[model]
    # created_model.apply(weight_init)
    created_model = created_model.cuda()
    return created_model

# 网络参数初始化
def weight_init(m):
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # torch.manual_seed(7)   # 随机种子，是否每次做相同初始化赋值
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        # m中的 weight，bias 其实都是 Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

