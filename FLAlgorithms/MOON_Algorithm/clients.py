import torch
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..' + '/' + '..'))
import copy
import torch.optim as optim
import torch.nn as nn
from data.generate_niid_dirichlet import Generate_niid_dirichelet
import numpy as np


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
        client = ClientMOON(args, model, i)
        
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


class ClientMOON(object):
    def __init__(self, args, model, i):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.cos=torch.nn.CosineSimilarity(dim=-1)
        self.previous_model = copy.deepcopy(model).cuda().eval()
        self.glob_model = copy.deepcopy(model).cuda().eval()
        self.temperature, self.mu = 0.5, 0.1

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
        print('Training on client #{}'.format(self.client_id))
        self.update(server_broadcast_dict)
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
                loss = loss_sup + loss_con
                loss.backward()
                self.optimizer.step()
                
        self.adjust_learning_rate(self.optimizer, glob_iter, decay=0.998, init_lr=args.learning_rate, lr_decay_epoch=1)

        # update previous model 
        self.previous_model.load_state_dict(self.model.state_dict())

        self.upload_dict = {"params":self.model.state_dict()}

        return self.upload_dict

    def update(self, server_broadcast_dict):
        if 'params_sum' in server_broadcast_dict.keys():
            self.model.load_state_dict(copy.deepcopy(server_broadcast_dict["params_sum"]))
            # update global model
            self.glob_model.load_state_dict(copy.deepcopy(server_broadcast_dict["params_sum"]))
