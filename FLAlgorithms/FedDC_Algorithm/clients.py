import torch
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..' + '/' + '..'))
import copy
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data.generate_niid_dirichlet import Generate_niid_dirichelet
from collections import OrderedDict


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
        client = ClientFedDC(args, model, i)
        
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


class ClientFedDC(object):
    def __init__(self, args, model, i):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.local_drift,self.local_gradient_drift,self.glob_gradient_drift = OrderedDict(),OrderedDict(),OrderedDict()

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
        print('Training on client #{}'.format(self.client_id))
        if glob_iter>0:
            self.update(server_broadcast_dict)
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

                if glob_iter>0 and (state_update_diff is not None):
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

        self.upload_dict = {"params":self.model.state_dict(), 'local_gradient_drift':self.local_gradient_drift}

        return self.upload_dict

    def update(self, server_broadcast_dict):
        if 'params_sum' in server_broadcast_dict.keys():
            self.model.load_state_dict(copy.deepcopy(server_broadcast_dict["params_sum"]))
            self.glob_gradient_drift = server_broadcast_dict["glob_gradient_drift"]