import copy
import torch
import numpy as np
from collections import OrderedDict

class ServerFedProx(object):
    def __init__(self, args, model, seed):
        self.num_clients = args.num_clients
        self.selected_num = args.selected_num
        self.args = args
        self.model = copy.deepcopy(model)
        self.broadcast_dict={}
        self.partial_sharing = args.partial_sharing
        
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
            encrypted_chunks = copy.deepcopy(c_params_list[0])
            for i,c_chunks in enumerate(c_params_list):
                if i>0:
                    for chunk_id, chunk in enumerate(c_chunks):
                        encrypted_chunks[chunk_id]+=copy.deepcopy(chunk)
            self.broadcast_dict["encrypted_sum"] = (copy.deepcopy(encrypted_chunks),c_upload_list[0]['c_encrypted_params'][1])
        else:
            c_params_list = [c_upload['c_encrypted_params'] for c_upload in c_upload_list]
            encrypted_sum = copy.deepcopy(c_params_list[0])
            for i,c_params in enumerate(c_params_list):
                if i>0:
                    encrypted_sum += copy.deepcopy(c_params)
            self.broadcast_dict["encrypted_sum"] = copy.deepcopy(encrypted_sum)

        for c_upload in c_upload_list:
            if 'gen_model' in c_upload.keys():
                self.broadcast_dict['gen_model'] = c_upload['gen_model']
            else:
                pass
        return self.broadcast_dict 