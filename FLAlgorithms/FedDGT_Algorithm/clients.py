
import torch
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..' + '/' + '..'))
import copy
import collections
import torch.optim as optim
import torch.nn as nn
from model_utils.create_model import create_generative_model
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
        client = ClientFedDGT(args, model, i)
        
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


class ClientFedDGT(object):
    def __init__(self, args, model, i):
        self.client_id = i
        self.model = copy.deepcopy(model)
        self.glob_model = copy.deepcopy(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        self.generative_alpha = 10
        model_parameters = self.model.state_dict()
        self.model_parameters_dict = collections.OrderedDict() 
        for key, value in model_parameters.items():
            self.model_parameters_dict[key] = torch.numel(value), value.shape
        self.criterion = nn.CrossEntropyLoss()
        self.record_layer_list = None
        self.upload={}
        # generative model
        self.generative_model = create_generative_model(dataset=args.dataset,model=args.model).cuda()
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=3e-4, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)

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
        print('Training on client #{}'.format(self.client_id))
        self.upload={}
        if glob_iter>0:
            self.update(args, server_broadcast_dict)
        self.model.cuda()
        self.model.train()
        for epoch in range(args.local_epochs):
            for i in range(args.K):
                if glob_iter >= args.warm_rounds:
                    # real local dataset
                    self.optimizer.zero_grad()
                    samples =self.get_next_train_batch(count_labels=True)
                    inputs, targets = samples['X'].cuda(), samples['y'].cuda()
                    _, output = self.model(inputs)
                    real_loss = nn.CrossEntropyLoss()(output, targets)
                    
                    # generated data from generator
                    generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    sampled_y=np.random.choice(self.labels, args.gen_batch_size)
                    sampled_y=torch.tensor(sampled_y).cuda()
                    gen_result=self.generative_model(sampled_y, latent_layer_idx=-1)
                    gen_output=gen_result['output'] # latent representation when latent = True, x otherwise
                    
                    _,user_output_logp =self.model(gen_output, start_layer_idx=-1)
                    teacher_loss =  generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    gen_ratio = args.gen_batch_size/args.batch_size
                    teacher_loss = gen_ratio*teacher_loss
                    teacher_loss.backward()
                    real_loss.backward()
                    self.optimizer.step()
                else:
                    # real local dataset
                    self.optimizer.zero_grad()
                    samples =self.get_next_train_batch(count_labels=True)
                    inputs, targets = samples['X'].cuda(), samples['y'].cuda()
                    _, output = self.model(inputs)
                    real_loss = nn.CrossEntropyLoss()(output, targets)
                    real_loss.backward()
                    self.optimizer.step()

        self.adjust_learning_rate(self.optimizer, glob_iter, decay=0.998, init_lr=args.learning_rate, lr_decay_epoch=1)

        self.upload = {"params":self.model.state_dict()}

        if glob_iter >= args.warm_rounds:
            # train generative model
            if self.client_id==server_broadcast_dict['train_gen_id']:
                self.train_generator(batch_size=args.training_gen_batch, steps=args.training_gen_step)
        else:
            self.upload['gen_model'] = self.generative_model.state_dict()

        return self.upload
        
    def train_generator(self, batch_size=20, epoches=1, latent_layer_idx=-1, verbose=True, steps=5):
        TEACHER_LOSS, DIVERSITY_LOSS,  = 0, 0
        
        def update_generator_(n_iters, teacher_model, TEACHER_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            teacher_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y=np.random.choice(self.labels, batch_size)
                y_input=torch.tensor(y).cuda()
                # feed to generator
                gen_result=self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                gen_output, eps=gen_result['output'], gen_result['eps']
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                # get teacher loss
                teacher_loss=0
                _,global_output_logp_=teacher_model(gen_output, start_layer_idx=latent_layer_idx)
                teacher_loss_=torch.mean(self.generative_model.crossentropy_loss(global_output_logp_, y_input))
                teacher_loss=teacher_loss_
                
                loss=teacher_loss + diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += teacher_loss
                DIVERSITY_LOSS += 1 * diversity_loss
            return TEACHER_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, DIVERSITY_LOSS=update_generator_(
                steps, self.glob_model, TEACHER_LOSS, DIVERSITY_LOSS)  # 这个时候的self.model是全局模型

        TEACHER_LOSS = TEACHER_LOSS / (5 * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS / (5 * epoches)
        info="Generator: Teacher Loss= {:.4f}, Diversity Loss = {:.4f}, ".format(TEACHER_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.upload['gen_model'] = self.generative_model.state_dict()

    def update(self, args, server_broadcast_dict):
        if 'params_sum' in server_broadcast_dict.keys():
            self.generative_model.load_state_dict(server_broadcast_dict['gen_model'])
            self.model.load_state_dict(copy.deepcopy(server_broadcast_dict["params_sum"]))
            self.glob_model.load_state_dict(copy.deepcopy(server_broadcast_dict["params_sum"]))

        