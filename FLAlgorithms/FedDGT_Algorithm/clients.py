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
from model_utils.create_model import create_generative_model
import numpy as np
from data.generate_niid_dirichlet import Generate_niid_dirichelet
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
        client = ClientFedDGT(args, model, i, pk, sk)
        
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
        client = ClientFedDGT(args, model, i, pk, sk)
        client.set_bias(bias, pref[i], partition_size[i])
        client_data = loader.get_partition(client)
        client.set_data(client_data, args.batch_size, loader.labels)
        clients.append(client)
    return testset, clients


class ClientFedDGT(object):
    def __init__(self, args, model, i, pk, sk):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.glob_model = copy.deepcopy(model)
        if args.dataset=='imdb':
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
            # self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.generative_alpha = 10
        self.generative_beta = 10
        model_parameters = self.model.state_dict()  # 提取网络参数
        self.model_parameters_dict = collections.OrderedDict() 
        for key, value in model_parameters.items():
            self.model_parameters_dict[key] = torch.numel(value), value.shape
        self.criterion = nn.CrossEntropyLoss()
        self.record_layer_list = None
        self.upload={}
        # generative model
        self.generative_model = create_generative_model(dataset=args.dataset,model=args.model).cuda()
        self.gen_parameters_dict = collections.OrderedDict() 
        for key, value in self.generative_model.state_dict().items():
            self.gen_parameters_dict[key] = torch.numel(value), value.shape
        
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=3e-4, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)

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
    
    def train(self, args, glob_iter, server_broadcast_dict):
        from HEUtils.cuda_test import KeyGen, Enc, Dec
        
        print('Training on client #{}'.format(self.client_id))
        self.upload={}
        if glob_iter>0:
            self.update(args, server_broadcast_dict)
        temp_model = copy.deepcopy(self.model)
        self.model.cuda()
        self.model.train()
        for epoch in range(args.local_epochs):
            for i in range(args.K):
                # real local dataset
                self.optimizer.zero_grad()
                samples =self.get_next_train_batch(count_labels=True)
                inputs, targets = samples['X'].cuda(), samples['y'].cuda()
                # inputs, targets = inputs.cuda(),targets.cuda()
                _, output = self.model(inputs)
                real_loss = nn.CrossEntropyLoss()(output, targets)
                
                # generated data from generator
                generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                generative_beta=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                sampled_y=np.random.choice(self.labels, args.gen_batch_size)
                sampled_y=torch.tensor(sampled_y).cuda()
                gen_result=self.generative_model(sampled_y, latent_layer_idx=-1)
                gen_output=gen_result['output'] # latent representation when latent = True, x otherwise
                
                _,user_output_logp =self.model(gen_output, start_layer_idx=-1)
                teacher_loss =  generative_alpha * torch.mean(
                    self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                )

                gen_ratio = args.gen_batch_size / args.batch_size
                # loss = real_loss + gen_ratio * teacher_loss
                teacher_loss.backward()
                # self.record_layer_list = [0,1,2,3,4,5,6,7]
                
                if self.record_layer_list is None:
                    self.record_layer_list=[]
                    for l,p in enumerate(self.model.parameters()):
                        if p.grad is None:
                            print('record layer:',l)
                            self.record_layer_list.append(l)
                
                real_loss.backward()
                for l,p in enumerate(self.model.parameters()):
                    if l in self.record_layer_list:
                        p.grad.data.mul_((args.gen_batch_size+args.batch_size)/args.batch_size)
                
                # loss = real_loss + gen_ratio * teacher_loss
                # loss.backward()
                self.optimizer.step()

        self.adjust_learning_rate(self.optimizer, glob_iter, decay=0.998, init_lr=args.learning_rate, lr_decay_epoch=1)
        sum, gen_sum = 0, 0
        for k, v in self.model.state_dict().items():
            sum += v.numel()
        for k, v in self.generative_model.state_dict().items():
            gen_sum += v.numel()
            
        gen_model = None
        # train generative model
        if self.client_id==server_broadcast_dict['train_gen_id']:
            gen_model = self.train_generator()
            
        if gen_model is not None:
            gen_params_list = []
            for k, v in gen_model.items():
                gen_params_list.append(copy.deepcopy(v).view(-1))
            
            if args.chunk:
                gen_params = torch.cat(gen_params_list, 0)
                chunks = math.ceil(len(gen_params) / 65536)
                chun = gen_params.chunk(chunks, 0)
                gen_chun_size_list = [i.numel() for i in chun]
                client_encrypted_gen_list = []
                for chun_p in chun:
                    p = ((chun_p + self.bound) * 2 ** self.prec).long().cuda()
                    client_encrypted_gen = Enc(self.pk, p)    # 加密梯度
                    client_encrypted_gen_list.append(copy.deepcopy(client_encrypted_gen))
                self.upload['c_encrypted_gen'] = (client_encrypted_gen_list, gen_chun_size_list)
            else:
                gen_params = ((torch.cat(gen_params_list, 0) + self.bound) * 2 ** self.prec).long().cuda()
                client_encrypted_gen = Enc(self.pk, gen_params)
                self.upload['c_encrypted_gen'] = client_encrypted_gen

        # upload dict with HE
        params_modules = list(self.model.named_parameters())
        params_list = []
        for params_module in params_modules:
            name, params = params_module
            params_list.append(copy.deepcopy(params.data).view(-1))

        if args.chunk:
            res = torch.cat(params_list, 0)
            chunks = math.ceil(len(res) / 65536)
            chun = res.chunk(chunks, 0)
            chun_size_list = [i.numel() for i in chun]
            client_encrypted_params_list = []
            for chun_p in chun:
                p = ((chun_p + self.bound) * 2 ** self.prec).long().cuda()
                client_encrypted_params = Enc(self.pk, p)    # 加密梯度
                client_encrypted_params_list.append(copy.deepcopy(client_encrypted_params))
            self.upload['c_encrypted_params'] = (client_encrypted_params_list,chun_size_list)
        else:
            params = ((torch.cat(params_list, 0) + self.bound) * 2 ** self.prec).long().cuda()
            client_encrypted_params = Enc(self.pk, params)    # 加密梯度
            self.upload['c_encrypted_params'] = client_encrypted_params

        return self.upload, temp_model
    
    def decode(self, args, encrypted_sum, encrypted_gen, selected_num):
        from HEUtils.cuda_test import KeyGen, Enc, Dec
        if args.chunk:
            data_sum_list, gen_sum_list = [], []
            encrypted_sum_chunks, chun_size_list = encrypted_sum
            encrypted_gen_chunks, gen_chun_list = encrypted_gen
            for i, encrypted_chunk in enumerate(encrypted_sum_chunks):
                data_sum = Dec(self.sk, encrypted_chunk).float() / (2 ** self.prec) / selected_num - self.bound
                data_sum_list.append(copy.deepcopy(data_sum[0:chun_size_list[i]]))
                gen_sum = Dec(self.sk, encrypted_gen_chunks[i]).float() / (2 ** self.prec) / selected_num - self.bound
                gen_sum_list.append(copy.deepcopy(gen_sum[0:gen_chun_list[i]]))
            decode_sum = torch.cat(data_sum_list, 0).cuda()
            decode_gen = torch.cat(gen_sum_list, 0).cuda()
        else:
            decode_sum = Dec(self.sk, encrypted_sum).float() / (2 ** self.prec) / selected_num - self.bound
            # decode_gen = Dec(self.sk, encrypted_gen).float() / (2 ** self.prec) / selected_num - self.bound
            decode_gen = Dec(self.sk, encrypted_gen).float() / (2 ** self.prec) - self.bound

        ind = 0
        client_data_dict = dict()
        for key in self.model_parameters_dict:
            params_size, params_shape = self.model_parameters_dict[key]
            client_data_dict[key] = decode_sum[ind : ind + params_size].reshape(params_shape)
            ind += params_size
            
        ind = 0
        client_gen_dict = dict()
        for key in self.gen_parameters_dict:
            params_size, params_shape = self.gen_parameters_dict[key]
            client_gen_dict[key] = decode_gen[ind : ind + params_size].reshape(params_shape)
            ind += params_size
        self.generative_model.load_state_dict(client_gen_dict)

        # 加载新的模型参数
        params_modules_server = self.model.named_parameters()
        for params_module in params_modules_server:
            name, params = params_module
            params.data = client_data_dict[name]  # 用字典中存储的子模型的梯度覆盖网络中的参数梯度

        self.model.load_state_dict(copy.deepcopy(client_data_dict))
        self.glob_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.global_model_params = copy.deepcopy(self.model.state_dict())
        
    def train_generator(self, batch_size=20, epoches=1, latent_layer_idx=-1, verbose=True):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        def update_generator_(n_iters, teacher_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            teacher_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y=np.random.choice(self.labels, batch_size)
                y_input=torch.tensor(y).cuda()
                ## feed to generator
                gen_result=self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss=0
                _,global_output_logp_=teacher_model(gen_output, start_layer_idx=latent_layer_idx)
                teacher_loss_=torch.mean(self.generative_model.crossentropy_loss(global_output_logp_, y_input))
                teacher_loss=teacher_loss_
                
                loss=1 * teacher_loss + 1 * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += 1 * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += -1.  #(torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += 1 * diversity_loss#(torch.mean(diversity_loss.double())).item()
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                5, self.glob_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)  # 这个时候的self.model是全局模型

        TEACHER_LOSS = TEACHER_LOSS / (5 * epoches)
        STUDENT_LOSS = STUDENT_LOSS / (5 * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS / (5 * epoches)
        info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        # self.generative_lr_scheduler.step()
        # self.upload['gen_model'] = self.generative_model.state_dict()
        return copy.deepcopy(self.generative_model.state_dict())

    def update(self, args, server_broadcast_dict):
        # without HE
        # if 'params_sum' in server_broadcast_dict.keys():
        #     self.generative_model.load_state_dict(server_broadcast_dict['gen_model'])
        #     self.model.load_state_dict(copy.deepcopy(server_broadcast_dict["params_sum"]))
        #     self.glob_model.load_state_dict(copy.deepcopy(server_broadcast_dict["params_sum"]))

        # with HE
        # self.generative_model.load_state_dict(server_broadcast_dict['gen_model'])
        encrypted_sum = server_broadcast_dict['encrypted_sum']
        encrypted_gen = server_broadcast_dict['encrypted_gen']
        selected_num = server_broadcast_dict['selected_num']
        self.decode(args, encrypted_sum, encrypted_gen, selected_num)
        