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
        client = ClientFedScaf(args, model, i, pk, sk)
        
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
        client = ClientFedScaf(args, model, i, pk, sk)
        client.set_bias(bias, pref[i], partition_size[i])
        client_data = loader.get_partition(client)
        client.set_data(client_data, args.batch_size, loader.labels)
        clients.append(client)
    return testset, clients

class ClientFedScaf(object):
    def __init__(self, args, model, i, pk, sk):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.upload = {}
        self.optimizer = FedScafOptimizer(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4,momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        model_parameters = self.model.state_dict()  # 提取网络参数
        self.model_parameters_dict = collections.OrderedDict() 
        for key, value in model_parameters.items():
            self.model_parameters_dict[key] = torch.numel(value), value.shape
        self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.delta_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

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
        from HEUtils.cuda_test import KeyGen, Enc, Dec

        print('Training on client #{}'.format(self.client_id))
        if 'params_sum' in server_broadcast_dict.keys():
            self.update(server_broadcast_dict)
        temp_model = copy.deepcopy(self.model)
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
            self.controls[i] = self.controls[i] - self.server_controls[i] + (p.data - temp[i]) / (cnt * args.learning_rate)

        # upload dict with HE
        params_modules = list(self.model.named_parameters())
        params_list, controls_list = [], []
        for i, params_module in enumerate(params_modules):
            name, params = params_module
            params_list.append(copy.deepcopy(params.data).view(-1))
            controls_list.append(copy.deepcopy(self.controls[i]).view(-1))

        if args.chunk:
            res = torch.cat(params_list, 0)
            con_res = torch.cat(controls_list, 0)
            chunks = math.ceil(len(res) / 65535)
            chun = res.chunk(chunks,0)
            con_chun = con_res.chunk(chunks,0)
            chun_size_list = [i.numel() for i in chun]
            client_encrypted_params_list, client_enc_controls_list = [], []
            for i, chun_p in enumerate(chun):
                p = ((chun_p + self.bound) * 2 ** self.prec).long().cuda()
                q = ((con_chun[i] + self.bound) * 2 ** self.prec).long().cuda()
                client_encrypted_params = Enc(self.pk, p)    # 加密梯度
                client_encrypted_controls = Enc(self.pk, q) 
                client_encrypted_params_list.append(copy.deepcopy(client_encrypted_params))
                client_enc_controls_list.append(copy.deepcopy(client_encrypted_controls))
            self.upload['c_encrypted_params'] = (client_encrypted_params_list,chun_size_list)
            self.upload['c_encrypted_controls'] = client_enc_controls_list
        else:
            params = ((torch.cat(params_list, 0) + self.bound) * 2 ** self.prec).long().cuda()
            controls = ((torch.cat(controls_list, 0) + self.bound) * 2 ** self.prec).long().cuda()
            client_encrypted_params = Enc(self.pk, params)    # 加密梯度
            client_encrypted_controls = Enc(self.pk, controls)
            self.upload['c_encrypted_params'] = client_encrypted_params
            self.upload['c_encrypted_controls'] = client_encrypted_controls

        # self.upload['c_controls'] = self.controls
        return self.upload, temp_model

    def decode(self, args, encrypted_sum, encrypted_controls, selected_num):
        from HEUtils.cuda_test import KeyGen, Enc, Dec
        if args.chunk:
            data_sum_list, control_sum_list = [], []
            encrypted_sum_chunks, chun_size_list = encrypted_sum
            encrypted_control_chunks = encrypted_controls
            for i, encrypted_chunk in enumerate(encrypted_sum_chunks):
                data_sum = Dec(self.sk, encrypted_chunk).float() / (2 ** self.prec) / selected_num - self.bound
                data_sum_list.append(copy.deepcopy(data_sum[0:chun_size_list[i]]))
                control_sum = Dec(self.sk, encrypted_control_chunks[i]).float() / (2 ** self.prec) / selected_num - self.bound
                control_sum_list.append(copy.deepcopy(control_sum[0: chun_size_list[i]]))
            decode_sum = torch.cat(data_sum_list, 0).cuda()
            decode_controls = torch.cat(control_sum_list, 0).cuda()
        else:
            decode_sum = Dec(self.sk, encrypted_sum).float() / (2 ** self.prec) / selected_num - self.bound
            decode_controls = Dec(self.sk, encrypted_controls).float() / (2 ** self.prec) / selected_num - self.bound

        ind = 0
        client_data_dict, client_control_list = dict(), []
        for i, key in enumerate(self.model_parameters_dict):
            params_size, params_shape = self.model_parameters_dict[key]
            client_data_dict[key] = decode_sum[ind : ind + params_size].reshape(params_shape)
            client_control_list[i] = decode_controls[ind : ind + params_size].reshape(params_shape)
            ind += params_size

        # 加载新的模型参数
        params_modules_server = self.model.named_parameters()
        for params_module in params_modules_server:
            name, params = params_module
            params.data = client_data_dict[name]  # 用字典中存储的子模型的梯度覆盖网络中的参数梯度
        self.model.load_state_dict(copy.deepcopy(client_data_dict))
        self.global_model_params = copy.deepcopy(self.model.state_dict())
        self.controls = client_control_list
        

    def update(self, args, server_broadcast_dict):
        # avg_controls = server_broadcast_dict["avg_controls"]
        # self.server_controls = copy.deepcopy(avg_controls)

        encrypted_sum = server_broadcast_dict['encrypted_sum']
        encrypted_controls = server_broadcast_dict["encrypted_controls_sum"]
        selected_num = server_broadcast_dict['selected_num']
        self.decode(args, encrypted_sum, encrypted_controls, selected_num)


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
                # d_p = p.grad.data + c.data - ci.data
                
                # d_p = p.grad.data
                p.data = p.data -group['lr'] * (d_p+ c.data - ci.data)

        return loss
