import torch
import torch.nn as nn

GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim, gan_channels
    'cifar10': (1024, 1024, 3, 10, 128, 64),
    'mnist': (256, 84, 1, 10, 32, 32),
    'fashionmnist': (256, 84, 1, 10, 32, 32),
    'fashionmnist-1': (128, 84, 1, 10, 32, 32),
    'fashionmnist-2': (64, 84, 1, 10, 32, 32),
    'fashionmnist-3': (64, 84, 1, 10, 64, 32),
    'fashionmnist-4': (64, 84, 1, 10, 16, 32),
    'emnist': (256, 128, 1, 47, 32, 32),
    'svhn': (256, 84, 1, 10, 64, 32),
    'covtype': (32, 8, 1, 2, 32, 32),
    'rcv1': (32, 8, 1, 2, 32, 32),
}

class Generator(nn.Module):
    def __init__(self, dataset='mnist', model='cnn', embedding=False, latent_layer_idx=-1):
        super(Generator, self).__init__()
        print("Dataset {}".format(dataset))
        self.embedding = embedding
        self.dataset = dataset
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim, self.gan_channels = GENERATORCONFIGS[dataset]
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False) # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim)).cuda() # sampling from Gaussian
        if verbose:
            result['eps'] = eps
        if self.embedding: 
            y_input = self.embedding_layer(labels).cuda()
        else:
            y_input = torch.FloatTensor(batch_size, self.n_class).cuda()    #(32,10)
            y_input.zero_()
            y_input.scatter_(1, labels.view(-1,1).cuda(), 1)
        z = torch.cat((eps, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
