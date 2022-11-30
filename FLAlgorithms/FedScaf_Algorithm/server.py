import copy
import torch
import numpy as np
from collections import OrderedDict


class ServerFedScaf(object):
    def __init__(self, args, model, seed):
        self.num_clients = args.num_clients
        self.model = copy.deepcopy(model)
        self.broadcast_dict = {}
        self.args = args
        self.partial_sharing = args.partial_sharing
        self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        
    def select_clients(self, epoch, selected_num):
        '''selects selected_num clients from all clients.
        Return:
            list of selected client ids
        '''
        self.broadcast_dict['selected_num'] = selected_num
        if selected_num==self.num_clients:
            print("All clients are selected")
            client_ids = [i for i in range(self.num_clients)]
            return client_ids
        
        selected_num = min(selected_num, self.num_clients)
        client_ids = np.random.choice(range(self.num_clients), selected_num, replace=False)
        return client_ids

    def aggregate(self, c_upload_list):
        if self.args.chunk:
            c_params_list = [c_upload['c_encrypted_params'][0] for c_upload in c_upload_list]
            c_controls_list = [c_upload['c_encrypted_controls'] for c_upload in c_upload_list]
            encrypted_chunks = copy.deepcopy(c_params_list[0])
            encrypted_controls = copy.deepcopy(c_controls_list[0])
            for i,c_chunks in enumerate(c_params_list):
                if i>0:
                    for chunk_id, chunk in enumerate(c_chunks):
                        encrypted_chunks[chunk_id] += copy.deepcopy(chunk)
                        encrypted_controls[chunk_id] += copy.deepcopy(c_controls_list[i][chunk_id])
            self.broadcast_dict["encrypted_sum"] = (copy.deepcopy(encrypted_chunks), c_upload_list[0]['c_encrypted_params'][1])
            self.broadcast_dict["encrypted_controls_sum"] = copy.deepcopy(encrypted_controls)
        else:
            c_params_list = [c_upload['c_encrypted_params'] for c_upload in c_upload_list]
            c_controls_list = [c_upload['c_encrypted_controls'] for c_upload in c_upload_list]
            encrypted_sum = copy.deepcopy(c_params_list[0])
            encrypted_controls = copy.deepcopy(c_controls_list[0])
            for i,c_params in enumerate(c_params_list):
                if i>0:
                    encrypted_sum += copy.deepcopy(c_params)
                    encrypted_controls += copy.deepcopy(c_controls_list[i])
            self.broadcast_dict["encrypted_sum"] = copy.deepcopy(encrypted_sum)
            self.broadcast_dict["encrypted_controls_sum"] = copy.deepcopy(encrypted_controls)

        # get avg controls
        # c_controls_list = [c_upload['c_controls'] for c_upload in c_upload_list]
        # avg_controls = [torch.zeros(x.size()).cuda() for x in c_controls_list[0]]
        # for i,c_controls in enumerate(c_controls_list):
        #     for k, l_controls in enumerate(c_controls):
        #         avg_controls[k] += (1/len(c_controls_list)) * l_controls
        
        # self.broadcast_dict["avg_controls"] = avg_controls
    
        return self.broadcast_dict