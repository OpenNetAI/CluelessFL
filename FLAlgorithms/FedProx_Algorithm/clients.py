class PublicKey:
    def __init__(self, A, P, n, s):
        self.A = A
        self.P = P
        self.n = n
        self.s = s

    def __repr__(self):
        return 'PublicKey({}, {}, {}, {})'.format(self.A, self.P, self.n, self.s)

import torch
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..' + '/' + '..'))
import random
import copy
import collections
import torch.optim as optim
import torch.nn as nn
sys.path.append("./HEUtils")
import numpy as np
import math
from data.generate_niid_dirichlet import Generate_niid_dirichelet

def generate_niid_dirichelet_Clients(args, model, pk, sk):
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
        client = ClientFedProx(args, model, i, pk, sk)
        
        data = user_data[user]
        datax = data['x']
        datay = data['y']
        
        client_data=[]

        for j in range(datax.shape[0]):
            if args.dataset == "imdb":
                client_data.append((torch.LongTensor(datax[j]),datay[j]))
            else:
                client_data.append((torch.tensor(datax[j]),datay[j]))
        
        shuffled_array = np.array(client_data)
        np.random.shuffle(shuffled_array)
        client_data = shuffled_array.tolist()
        client.set_data(client_data, args.batch_size, labels)
        clients.append(client)
    return testset, clients


def generate_clients(args, model, pk, sk):
    data_generator = load_data.DataGenerator()
    # Generate local data by label
    _,testset = data_generator.generate(args)       # generate grouped dataset
    unique_labels = data_generator.labels
    loader = load_data.BiasLoader(data_generator)

    # config local data
    if isinstance(args.p_size, list):
        start, stop = args.p_size[0], args.p_size[-1]
        partition_size = [random.randint(start, stop) for _ in range(args.num_clients)]
    else:
        partition_size = [args.p_size for _ in range(args.num_clients)]
    # pref = [random.choice(unique_labels) for _ in range(args.num_clients)]
    pref = unique_labels * (int(args.num_clients/len(unique_labels)))
    random.shuffle(pref)
    bias = args.bias
    
    # generate clients
    clients = []
    for i in range(args.num_clients):
        client = ClientFedProx(args, model, i, pk, sk)
        client.set_bias(bias, pref[i], partition_size[i])
        client_data = loader.get_partition(client)
        client.set_data(client_data, args.batch_size, loader.labels)
        clients.append(client)
    return testset, clients

class ClientFedProx(object):
    def __init__(self, args, model, i, pk, sk):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.optimizer = FedProxOptimizer(self.model.parameters(), lr=args.learning_rate, momentum=0.9)
        self.star_params_list = copy.deepcopy(list(self.model.parameters()))
        self.generative_alpha = 10
        self.generative_beta = 10
        self.criterion = nn.CrossEntropyLoss()
        self.upload = {}
        model_parameters = self.model.state_dict()  # 提取网络参数
        self.model_parameters_dict = collections.OrderedDict() 
        for key, value in model_parameters.items():
            self.model_parameters_dict[key] = torch.numel(value), value.shape

        # HE settings
        self.prec = 32
        self.bound = 2 ** 3
        self.pk = pk
        self.sk = sk

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

    def adjust_learning_rate(self, optim, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        for param_group in optim.param_groups:
            param_group['lr'] = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
            if epoch % 50 == 0:
                print('current learning rate:', param_group['lr'])
    
    def train(self,args, glob_iter, server_broadcast_dict):
        from HEUtils.cuda_test import KeyGen, Enc, Dec
        
        print('Training on client #{}'.format(self.client_id))
        if glob_iter > 0:
            self.update(args,server_broadcast_dict)
        temp_model = copy.deepcopy(self.model)
        self.model.cuda()
        self.model.train()
        self.star_params_list = copy.deepcopy(list(self.model.parameters()))
        for epoch in range(args.local_epochs):
            for i in range(args.K):
                # real local dataset
                self.optimizer.zero_grad()
                samples =self.get_next_train_batch(count_labels=True)
                inputs, targets = samples['X'].cuda(), samples['y'].cuda()
                _, output = self.model(inputs)
                real_loss = self.criterion(output, targets)
                real_loss.backward()
                self.optimizer.step(self.star_params_list)
                
        self.adjust_learning_rate(self.optimizer, glob_iter, decay=0.998, init_lr=args.learning_rate, lr_decay_epoch=1)

        # upload dict with HE
        params_modules = list(self.model.named_parameters())
        params_list = []
        for params_module in params_modules:
            name, params = params_module
            params_list.append(copy.deepcopy(params.data).view(-1))

        if args.chunk:
            res = torch.cat(params_list, 0)
            chunks = math.ceil(len(res) / 65536)
            chun = res.chunk(chunks,0)
            chun_size_list = [i.numel() for i in chun]
            client_encrypted_params_list = []
            for chun_p in chun:
                p = ((chun_p + self.bound) * 2 ** self.prec).long().cuda()
                client_encrypted_params = Enc(self.pk, p)  
                client_encrypted_params_list.append(copy.deepcopy(client_encrypted_params))
            self.upload['c_encrypted_params'] = (client_encrypted_params_list,chun_size_list)
        else:
            params = ((torch.cat(params_list, 0) + self.bound) * 2 ** self.prec).long().cuda()
            client_encrypted_params = Enc(self.pk, params)    # 加密梯度
            self.upload['c_encrypted_params'] = client_encrypted_params

        return self.upload, temp_model

    def decode(self, args, encrypted_sum, selected_num):
        from HEUtils.cuda_test import KeyGen, Enc, Dec
        if args.chunk:
            data_sum_list=[]
            encrypted_sum_chunks, chun_size_list = encrypted_sum
            for i,encrypted_chunk in enumerate(encrypted_sum_chunks):
                data_sum = Dec(self.sk, encrypted_chunk).float() / (2 ** self.prec) / selected_num - self.bound
                data_sum_list.append(copy.deepcopy(data_sum[0:chun_size_list[i]]))
            decode_sum = torch.cat(data_sum_list,0).cuda()
        else:
            decode_sum = Dec(self.sk, encrypted_sum).float() / (2 ** self.prec) / selected_num - self.bound

        ind = 0
        client_data_dict = dict()
        for key in self.model_parameters_dict:
            params_size, params_shape = self.model_parameters_dict[key]
            client_data_dict[key] = decode_sum[ind : ind + params_size].reshape(params_shape)
            ind += params_size

        # 加载新的模型参数
        params_modules_server = self.model.named_parameters()
        for params_module in params_modules_server:
            name, params = params_module
            params.data = client_data_dict[name]  # 用字典中存储的子模型的梯度覆盖网络中的参数梯度
        self.model.load_state_dict(copy.deepcopy(client_data_dict))
        self.global_model_params = copy.deepcopy(self.model.state_dict())
        

    def update(self, args, server_broadcast_dict):
        encrypted_sum = server_broadcast_dict['encrypted_sum']
        selected_num = server_broadcast_dict['selected_num']
        self.decode(args, encrypted_sum, selected_num)

class FedProxOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001, momentum=0.9):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr, lamda=lamda, mu=mu, momentum=momentum)
        super(FedProxOptimizer, self).__init__(params, defaults)

    def step(self, vstar, closure=None):
        loss=None
        if closure is not None:
            loss=closure
        for group in self.param_groups:
            momentum = group['momentum']
            for p, pstar in zip(group['params'], vstar):
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
                # w <=== w - lr * ( w'  + lambda * (w - w* ) + mu * w )
                # p.data=p.data - group['lr'] * (
                #             p.grad.data + group['lamda'] * (p.data - pstar.data.clone()) + group['mu'] * p.data)
                p.data=p.data - group['lr'] * (
                            d_p + group['lamda'] * (p.data - pstar.data.clone()) + group['mu'] * p.data)
        return group['params'], loss


class FedProxAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, lamda=0.1, mu=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr, lamda=lamda, mu=mu, betas=betas, eps=eps,weight_decay=weight_decay)
        super(FedProxAdam, self).__init__(params, defaults)

    def step(self, vstar, closure=None):
        loss=None
        if closure is not None:
            loss=closure
        for group in self.param_groups:
            for p, pstar in zip(group['params'], vstar):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                
                # # nwy
                # d_p = grad + group['lamda'] * (p.data - pstar.data.clone())

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data = p.data - step_size*(exp_avg/denom + group['lamda'] * (p.data - pstar.data.clone())\
                                                + group['mu']*p.data)

        return loss

