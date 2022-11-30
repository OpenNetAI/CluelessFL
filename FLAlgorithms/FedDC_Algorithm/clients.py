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
import copy
import collections
import torch.optim as optim
import torch.nn as nn
sys.path.append("./HEUtils")
import numpy as np
from data.generate_niid_dirichlet import Generate_niid_dirichelet
from collections import OrderedDict
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
        client = ClientFedDC(args, model, i, pk, sk)
        
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

class ClientFedDC(object):
    def __init__(self, args, model, i, pk, sk):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.upload = {}
        if args.dataset=='imdb':
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.local_drift,self.local_gradient_drift,self.glob_gradient_drift = OrderedDict(),OrderedDict(),OrderedDict()
        # self.local_drift = [torch.zeros_like(p.data) for p in self.model.parameters()]
        # self.local_gradient_drift = [torch.zeros_like(p.data) for p in self.model.parameters()]
        # self.glob_gradient_drift = [torch.zeros_like(p.data) for p in self.model.parameters()]

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
        if glob_iter>0:
            self.update(args, server_broadcast_dict)
        temp_model = copy.deepcopy(self.model)
        glob_model_params_dict = self.model.state_dict()
        global_model_param = None
        for key in glob_model_params_dict.keys():
            if not isinstance(global_model_param, torch.Tensor):
                global_model_param = glob_model_params_dict[key].clone().view(-1)
            else:
                global_model_param = torch.cat((global_model_param, glob_model_params_dict[key].clone().view(-1)), dim=0)
        
        if glob_iter>0:
            epoch_loss=0
            state_update_diff = None
            for key in self.local_gradient_drift.keys():
                if not isinstance(state_update_diff, torch.Tensor):
                    state_update_diff = (self.local_gradient_drift[key].clone()-self.glob_gradient_drift[key]).clone().view(-1)
                else:
                    state_update_diff = torch.cat((state_update_diff,(self.local_gradient_drift[key].clone()-self.glob_gradient_drift[key]).clone().view(-1)),dim=0)
            hist_i = None
            for key in self.local_drift.keys():
                if not isinstance(hist_i, torch.Tensor):
                    hist_i = self.local_drift[key].clone().view(-1)
                else:
                    hist_i = torch.cat((hist_i,self.local_drift[key].clone().view(-1)),dim=0)
        self.model.cuda()
        self.model.train()
        for epoch in range(args.local_epochs):
            for i in range(args.K):
                # real local dataset
                self.optimizer.zero_grad()
                samples =self.get_next_train_batch(count_labels=True)
                inputs, targets = samples['X'].cuda(), samples['y'].cuda()
                _, output = self.model(inputs)
                loss_sup = self.criterion(output, targets)

                if glob_iter>0:
                    # loss_cg
                    curr_params = None
                    model_state_dict = self.model.state_dict()
                    for key in model_state_dict.keys():
                        if not isinstance(curr_params, torch.Tensor):
                            curr_params = model_state_dict[key].clone().view(-1)
                        else:
                            curr_params = torch.cat((curr_params, model_state_dict[key].clone().view(-1)), dim=0)
                    
                    loss_cg = torch.sum(curr_params * state_update_diff)
                    
                    # loss_cp
                    alpha=0.1
                    loss_cp = alpha/2 * torch.sum((curr_params - (global_model_param - hist_i))*(curr_params - (global_model_param - hist_i)))
                    
                    # print(loss_cg.item(),loss_cp.item())
                    loss = loss_sup + loss_cg/(args.K*args.local_epochs)/args.learning_rate + loss_cp 
                else:
                    loss = loss_sup

                loss.backward()
                self.optimizer.step()
                
        self.adjust_learning_rate(self.optimizer, glob_iter, decay=0.998, init_lr=args.learning_rate, lr_decay_epoch=1)

        # update local drift
        local_model_params_dict = self.model.state_dict()
        for key in local_model_params_dict.keys():
            self.local_gradient_drift[key] = (local_model_params_dict[key]-glob_model_params_dict[key]).clone()
            if key not in self.local_drift.keys():
                self.local_drift[key] = self.local_gradient_drift[key].clone()
            else:
                self.local_drift[key] += self.local_gradient_drift[key].clone()
        # for i, local_param in enumerate(local_model_params):
        #     self.local_gradient_drift[i] = local_param.clone().detach()-glob_model_params[i].clone().detach()
        #     self.local_drift[i] += self.local_gradient_drift[i]

        # upload dict with HE
        params_modules = list(self.model.named_parameters())
        params_list, drift_list = [], []
        for i, params_module in enumerate(params_modules):
            name, params = params_module
            params_list.append(copy.deepcopy(params.data).view(-1))
        for key, value in self.local_gradient_drift.items():
            drift_list.append(copy.deepcopy(value).view(-1))

        if args.chunk:
            res = torch.cat(params_list, 0)
            drift_res = torch.cat(drift_list, 0)
            chunks = math.ceil(len(res) / 65535)
            chun = res.chunk(chunks,0)
            drift_chun = drift_res.chunk(chunks,0)
            chun_size_list = [i.numel() for i in chun]
            client_encrypted_params_list, client_encrypted_drifts_list = [], []
            enc_time, dec_time = 0, 0
            for i, chun_p in enumerate(chun):
                p = ((chun_p + self.bound) * 2 ** self.prec).long().cuda()
                q = ((drift_chun[i] + self.bound) * 2 ** self.prec).long().cuda()
                client_encrypted_params = Enc(self.pk, p)    # 加密梯度
                client_encrypted_drifts = Enc(self.pk, q)
                client_encrypted_params_list.append(copy.deepcopy(client_encrypted_params))
                client_encrypted_drifts_list.append(copy.deepcopy(client_encrypted_drifts))
            self.upload['c_encrypted_params'] = (client_encrypted_params_list,chun_size_list)
            self.upload['c_encrypted_drifts'] = client_encrypted_drifts_list
        else:
            params = ((torch.cat(params_list, 0) + self.bound) * 2 ** self.prec).long().cuda()
            drifts = ((torch.cat(drift_list, 0) + self.bound) * 2 ** self.prec).long().cuda()
            client_encrypted_params = Enc(self.pk, params)    # 加密梯度
            client_encrypted_drifts = Enc(self.pk, drifts)
            self.upload['c_encrypted_params'] = client_encrypted_params
            self.upload['c_encrypted_drifts'] = client_encrypted_drifts

        # self.upload['local_gradient_drift'] = self.local_gradient_drift

        return self.upload, temp_model
    
    def decode(self, args, encrypted_sum, encrypted_drifts, selected_num):
        from HEUtils.cuda_test import KeyGen, Enc, Dec
        if args.chunk:
            data_sum_list, drift_sum_list = [], []
            encrypted_sum_chunks, chun_size_list = encrypted_sum
            encrypted_drift_chunks = encrypted_drifts
            for i,encrypted_chunk in enumerate(encrypted_sum_chunks):
                data_sum = Dec(self.sk, encrypted_chunk).float() / (2 ** self.prec) / selected_num - self.bound
                data_sum_list.append(copy.deepcopy(data_sum[0:chun_size_list[i]]))
                drift_sum = Dec(self.sk, encrypted_drift_chunks[i]).float() / (2 ** self.prec) / selected_num - self.bound
                drift_sum_list.append(copy.deepcopy(drift_sum[0:chun_size_list[i]]))
            decode_sum = torch.cat(data_sum_list, 0).cuda()
            decode_drifts = torch.cat(drift_sum_list, 0).cuda()
        else:
            decode_sum = Dec(self.sk, encrypted_sum).float() / (2 ** self.prec) / selected_num - self.bound
            decode_drifts = Dec(self.sk, encrypted_drifts).float() / (2 ** self.prec) / selected_num - self.bound

        ind = 0
        client_data_dict, client_drift_dict = dict(), OrderedDict()
        for key in self.model_parameters_dict:
            params_size, params_shape = self.model_parameters_dict[key]
            client_data_dict[key] = decode_sum[ind : ind + params_size].reshape(params_shape)
            client_drift_dict[key] = decode_drifts[ind : ind + params_size].reshape(params_shape)
            ind += params_size

        # 加载新的模型参数
        params_modules_server = self.model.named_parameters()
        for params_module in params_modules_server:
            name, params = params_module
            params.data = client_data_dict[name]  # 用字典中存储的子模型的梯度覆盖网络中的参数梯度
        self.model.load_state_dict(copy.deepcopy(client_data_dict))
        self.global_model_params = copy.deepcopy(self.model.state_dict())
        self.glob_gradient_drift = client_drift_dict

    def update(self, args, server_broadcast_dict):
        # self.glob_gradient_drift = server_broadcast_dict["glob_gradient_drift"]
        
        encrypted_sum = server_broadcast_dict['encrypted_sum']
        encrypted_drifts = server_broadcast_dict["encrypted_drifts_sum"]
        selected_num = server_broadcast_dict['selected_num']
        self.decode(args, encrypted_sum, encrypted_drifts, selected_num)