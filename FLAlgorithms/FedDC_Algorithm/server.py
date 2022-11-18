from audioop import avg
import copy
import torch
import numpy as np
from collections import OrderedDict

class ServerFedDC(object):
    def __init__(self, args, model, seed):
        self.num_clients = args.num_clients
        self.selected_num = args.selected_num
        self.model = copy.deepcopy(model)
        self.params_sum = [p.data.clone() for p in self.model.parameters()]
        self.broadcast_dict = {}
        self.partial_sharing = args.partial_sharing
    
    def select_clients(self, epoch, selected_num):
        '''selects selected_num clients from all clients.
        Return:
            list of selected client ids
        '''
        if selected_num==self.num_clients:
            print("All clients are selected")
            client_ids = [i for i in range(self.num_clients)]
            return client_ids
        
        self.selected_num = min(selected_num, self.num_clients)
        client_ids = np.random.choice(range(self.num_clients), self.selected_num, replace=False)
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
        
        total_drift = copy.deepcopy(c_upload_list[0]['local_gradient_drift'])
        for i in range(1, len(c_upload_list)):
            for key in total_drift.keys():
                total_drift[key] += c_upload_list[i]['local_gradient_drift'][key]
        avg_drift = OrderedDict()
        for key in total_drift.keys():
            avg_drift[key] = total_drift[key] / len(c_upload_list)

        self.broadcast_dict = {"params_sum":avg_state_dict, "glob_gradient_drift":avg_drift}
