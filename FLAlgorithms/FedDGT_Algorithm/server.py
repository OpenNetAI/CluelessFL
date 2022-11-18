import copy
import numpy as np
import torch
from model_utils.create_model import create_generative_model
from collections import OrderedDict

class ServerFedDGT(object):
    def __init__(self, args, model, seed):
        self.num_clients = args.num_clients
        self.model = copy.deepcopy(model)
        self.selected_num = args.selected_num
        self.broadcast_dict={}
        self.train_gen_id = 0
        self.args = args
    
    def select_clients(self, epoch, selected_num, random_train_gen=True):
        '''selects selected_num clients from all clients.
        Return:
            list of selected client ids
        '''
        if selected_num==self.num_clients:
            print("All clients are selected")
            client_ids = [i for i in range(self.num_clients)]
        else:
            selected_num = min(selected_num, self.num_clients)
            client_ids = np.random.choice(range(self.num_clients), selected_num, replace=False)
        self.broadcast_dict['selected_num'] = selected_num
        self.selected_num = selected_num
        if random_train_gen==True:
            self.train_gen_id = np.random.choice(client_ids)
        self.broadcast_dict['train_gen_id'] = self.train_gen_id
        return client_ids
    
    def aggregate(self, c_upload_list):
        total_state_dict = copy.deepcopy(c_upload_list[0]['params'])
        for i in range(1, len(c_upload_list)):
            for key in total_state_dict.keys():
                total_state_dict[key] += c_upload_list[i]['params'][key]
        avg_state_dict = OrderedDict()
        for key in total_state_dict.keys():
            avg_state_dict[key] = total_state_dict[key] / len(c_upload_list)
        self.model.load_state_dict(avg_state_dict)
        self.model.eval()
        self.broadcast_dict = {"params_sum":avg_state_dict}
        for c_upload in c_upload_list:
            if 'gen_model' in c_upload.keys():
                self.broadcast_dict['gen_model'] = c_upload['gen_model']
            else:
                pass
        return self.broadcast_dict 

