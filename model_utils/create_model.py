from .lenet import LeNet
from .alexnet import AlexNet_CIFAR10
from .emnistcnn import EMNISTCNN
from .svhncnn import SVHNCNN
from .cmlp import CategoricalMLP
from .generator import Generator


def create_mlp_model(model, input_dim, hidden_dims, output_dim, num_classes=10):
    created_model = CategoricalMLP(input_dim, hidden_dims, output_dim).cuda()
    return created_model


def create_model(model, num_classes=10):
    created_model = {
        "lenet": LeNet().cuda(),
        "alexnet": AlexNet_CIFAR10().cuda(),
        "emnistcnn": EMNISTCNN().cuda(),
        "svhncnn": SVHNCNN().cuda(),
    }[model]
    created_model = created_model.cuda()
    return created_model


def create_generative_model(dataset, model='cnn', embedding=False):
    return Generator(dataset, model=model, embedding=embedding, latent_layer_idx=-1)

