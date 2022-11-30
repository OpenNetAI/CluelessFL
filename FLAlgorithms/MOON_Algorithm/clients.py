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
import torch.optim as optim
import torch.nn as nn
sys.path.append("./HEUtils")
from data.generate_niid_dirichlet import Generate_niid_dirichelet
import numpy as np
import collections
import math

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
        client = ClientMOON(args, model, i, pk, sk)
        
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
    _, testset = data_generator.generate(args)       # generate grouped dataset
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
        client = ClientMOON(args, model, i, pk, sk)
        client.set_bias(bias, pref[i], partition_size[i])
        client_data = loader.get_partition(client)
        client.set_data(client_data, args.batch_size, loader.labels)
        clients.append(client)
    return testset, clients

class ClientMOON(object):
    def __init__(self, args, model, i, pk, sk):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.upload = {}
        if args.dataset=='imdb':
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.cos=torch.nn.CosineSimilarity(dim=-1)
        self.previous_model = copy.deepcopy(model).cuda().eval()
        self.glob_model = copy.deepcopy(model).cuda().eval()
        self.temperature, self.mu = 0.5, 0.1

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
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size)
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
    
    def adjust_learning_rate(self, optim, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        for param_group in optim.param_groups:
            param_group['lr'] = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
            if epoch % 50 == 0:
                print('current learning rate:', param_group['lr'])
    
    def train(self,args, glob_iter, server_broadcast_dict):
        from HEUtils.cuda_test import KeyGen, Enc, Dec
        
        print('Training on client #{}'.format(self.client_id))
        if glob_iter > 0:
            self.update(args, server_broadcast_dict)
        temp_model = copy.deepcopy(self.model)
        self.model.cuda()
        self.model.train()
        for epoch in range(args.local_epochs):
            for i in range(args.K):
            # for i in range(1):
                # real local dataset
                self.optimizer.zero_grad()
                samples =self.get_next_train_batch(count_labels=True)
                inputs, targets = samples['X'].cuda(), samples['y'].cuda()
                prob, output = self.model(inputs)
                prob_glob, _ = self.glob_model(inputs)
                prob_prev, _ = self.previous_model(inputs)

                # similarity between current model and global model
                posi = self.cos(prob, prob_glob)
                logits = posi.reshape(-1,1)

                # similarity between current model and previous model
                nega = self.cos(prob, prob_prev)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                
                logits /= self.temperature
                labels = torch.zeros(inputs.size(0)).cuda().long()

                loss_con = self.mu * self.criterion(logits, labels)
                loss_sup = self.criterion(output, targets)
                # print(logits)
                # exit(0)
                loss = loss_sup + loss_con
                loss.backward()
                self.optimizer.step()
                
        self.adjust_learning_rate(self.optimizer, glob_iter, decay=0.998, init_lr=args.learning_rate, lr_decay_epoch=1)

        # update previous model 
        self.previous_model.load_state_dict(self.model.state_dict())

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
                client_encrypted_params = Enc(self.pk, p)  # 加密梯度 
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
