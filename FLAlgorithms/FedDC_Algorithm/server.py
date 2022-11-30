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
        # self.model_parameters = self.model.state_dict()
        self.broadcast_dict = {}
        self.args = args
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
        
        self.selected_num = min(selected_num, self.num_clients)
        client_ids = np.random.choice(range(self.num_clients), self.selected_num, replace=False)
        return client_ids
    
    def aggregate(self, c_upload_list):
        if self.args.chunk:
            c_params_list = [c_upload['c_encrypted_params'][0] for c_upload in c_upload_list]
            c_drifts_list = [c_upload['c_encrypted_drifts'] for c_upload in c_upload_list]
            encrypted_chunks = copy.deepcopy(c_params_list[0])
            encrypted_drifts = copy.deepcopy(c_drifts_list[0])
            for i,c_chunks in enumerate(c_params_list):
                if i>0:
                    for chunk_id, chunk in enumerate(c_chunks):
                        encrypted_chunks[chunk_id]+=copy.deepcopy(chunk)
                        encrypted_drifts[chunk_id] += copy.deepcopy(c_drifts_list[i][chunk_id])
            self.broadcast_dict["encrypted_sum"] = (copy.deepcopy(encrypted_chunks),c_upload_list[0]['c_encrypted_params'][1])
            self.broadcast_dict["encrypted_drifts_sum"] = copy.deepcopy(encrypted_drifts)
        else:
            c_params_list = [c_upload['c_encrypted_params'] for c_upload in c_upload_list]
            c_drifts_list = [c_upload['c_encrypted_drifts'] for c_upload in c_upload_list]
            encrypted_sum = copy.deepcopy(c_params_list[0])
            encrypted_drifts = copy.deepcopy(c_drifts_list[0])
            for i,c_params in enumerate(c_params_list):
                if i>0:
                    encrypted_sum += copy.deepcopy(c_params)
                    encrypted_drifts += copy.deepcopy(c_drifts_list[i])
            self.broadcast_dict["encrypted_sum"] = copy.deepcopy(encrypted_sum)
            self.broadcast_dict["encrypted_drifts_sum"] = copy.deepcopy(encrypted_drifts)
            
        # total_drift = copy.deepcopy(c_upload_list[0]['local_gradient_drift'])
        # for i in range(1, len(c_upload_list)):
        #     for key in total_drift.keys():
        #         total_drift[key] += c_upload_list[i]['local_gradient_drift'][key]
        # avg_drift = OrderedDict()
        # for key in total_drift.keys():
        #     avg_drift[key] = total_drift[key] / len(c_upload_list)

        # self.broadcast_dict["glob_gradient_drift"] = avg_drift

        return self.broadcast_dict
    
    # def init_params_sum(self):
    #     init_params_sum = []
    #     for p in self.model.parameters():
    #         init_params_sum.append(p.data.clone())
    #     return init_params_sum
