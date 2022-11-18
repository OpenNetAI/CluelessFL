import torch
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..' + '/' + '..'))
import copy
import collections
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data.generate_niid_dirichlet import Generate_niid_dirichelet


def generate_niid_dirichelet_Clients(args, model):
    trainsetdata_dict, testsetdata_dict, testset, labels= Generate_niid_dirichelet(args)
    print("Complete to generate dataset.")
    print("Generating Clients......")
    labels.sort()
    users = trainsetdata_dict['users']
    user_data = trainsetdata_dict['user_data']
    num_samples = trainsetdata_dict['num_samples']

    print("num_samples of clients: ", num_samples)
    
    clients = []
    for i, user in enumerate(users):
        client = ClientFedScaf(args, model, i)
        
        data = user_data[user]
        datax = data['x']
        datay = data['y']
        
        client_data=[]

        for j in range(datax.shape[0]):
            client_data.append((torch.tensor(datax[j]),datay[j]))
        
        shuffled_array = np.array(client_data)
        np.random.shuffle(shuffled_array)
        client_data = shuffled_array.tolist()
        client.set_data(client_data, args.batch_size, labels)
        clients.append(client)
    return testset, clients


class ClientFedScaf(object):
    def __init__(self, args, model, i):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.optimizer = FedScafOptimizer(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4,momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        model_parameters = self.model.state_dict()  # 提取网络参数
        self.model_parameters_dict = collections.OrderedDict() 
        for key, value in model_parameters.items():
            self.model_parameters_dict[key] = torch.numel(value), value.shape
        self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.delta_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.alpha=0.1

    # Set non-IID data configurations
    def set_bias(self, bias, pref, partition_size):
        self.pref = pref
        self.bias = bias
        self.partition_size = partition_size
    
    def set_data(self, data, batch_size, labels):
        # Download data
        self.data = self.download(data)
        self.labels = labels

        # Extract trainset, testset (if applicable)
        data = self.data
        self.trainset = data
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        self.iter_trainloader = iter(self.trainloader)
    
    # Server interactions
    def download(self, argv):
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv

    def adjust_learning_rate(self, optim, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        for param_group in optim.param_groups:
            param_group['lr'] = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
            if epoch % 50 == 0:
                print('current learning rate:', param_group['lr'])
    
    def get_next_train_batch(self, count_labels=True):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        return result
    
    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr
    
    def train(self,args, glob_iter, server_broadcast_dict):
        print('Training on client #{}'.format(self.client_id))
        if 'params_sum' in server_broadcast_dict.keys():
            self.update(server_broadcast_dict)
        self.model.cuda()
        self.model.train()
        g_model = copy.deepcopy(self.model)
        cnt=0
        for epoch in range(args.local_epochs):
            for i in range(args.K):
                # real local dataset
                self.optimizer.zero_grad()
                samples =self.get_next_train_batch(count_labels=True)
                inputs, targets = samples['X'].cuda(), samples['y'].cuda()
                _, output = self.model(inputs)
                real_loss = self.criterion(output, targets)
                real_loss.backward()
                self.optimizer.step(self.server_controls, self.controls)
                cnt+=1

        self.adjust_learning_rate(self.optimizer, glob_iter, decay=0.998, init_lr=args.learning_rate, lr_decay_epoch=1)

        # get controls
        temp = [torch.zeros_like(p.data) for p in self.model.parameters()]
        for i,p in enumerate(self.model.parameters()):
            temp[i] = p.data.clone()
        for i, p in enumerate(g_model.parameters()):
            self.controls[i] = self.controls[i] - self.server_controls[i] +\
                                            self.alpha*(p.data - temp[i]) / (cnt * args.learning_rate)

        # get params
        self.upload = {"params":self.model.state_dict(), 'c_controls':self.controls}
        return self.upload

    def update(self, server_broadcast_dict):
        if 'params_sum' in server_broadcast_dict.keys():
            params_sum, avg_controls = server_broadcast_dict["params_sum"],server_broadcast_dict["avg_controls"]
            self.model.load_state_dict(copy.deepcopy(params_sum))
            self.server_controls = copy.deepcopy(avg_controls)


class FedScafOptimizer(optim.Optimizer):
    def __init__(self, params, lr, weight_decay, momentum):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super(FedScafOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            momentum = group['momentum']
            for p,c, ci in zip(group['params'], server_controls, client_controls):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum!=0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data = p.data -group['lr'] * (d_p+ c.data - ci.data)

        return loss
