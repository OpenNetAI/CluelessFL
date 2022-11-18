import copy
import torch
import numpy as np
from collections import OrderedDict


class ServerFedScaf(object):
    def __init__(self, args, model, seed):
        self.num_clients = args.num_clients
        self.model = copy.deepcopy(model)
        self.broadcast_dict={}
        self.partial_sharing = args.partial_sharing
        self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        
    def select_clients(self, epoch, selected_num):
        '''selects selected_num clients from all clients.
        Return:
            list of selected client ids
        '''
        if selected_num==self.num_clients:
            print("All clients are selected")
            client_ids = [i for i in range(self.num_clients)]
            return client_ids
        
        selected_num = min(selected_num, self.num_clients)
        client_ids = np.random.choice(range(self.num_clients), selected_num, replace=False)
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

        # get avg controls
        c_controls_list = [c_upload['c_controls'] for c_upload in c_upload_list]
        avg_controls = [torch.zeros(x.size()).cuda() for x in c_controls_list[0]]
        for i,c_controls in enumerate(c_controls_list):
            for k, l_controls in enumerate(c_controls):
                avg_controls[k] += (1/len(c_controls_list)) * l_controls
        
        self.broadcast_dict = {"params_sum":avg_state_dict, "avg_controls":avg_controls}