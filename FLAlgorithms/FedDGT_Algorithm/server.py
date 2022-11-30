import copy
import re
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
        self.partial_sharing = args.partial_sharing
        self.generative_model = create_generative_model(dataset=args.dataset,model=args.model).cuda()
    
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
        if self.args.chunk:
            c_params_list = [c_upload['c_encrypted_params'][0] for c_upload in c_upload_list]
            encrypted_chunks = copy.deepcopy(c_params_list[0])
            for i,c_chunks in enumerate(c_params_list):
                if i>0:
                    for chunk_id, chunk in enumerate(c_chunks):
                        encrypted_chunks[chunk_id] += copy.deepcopy(chunk)
            self.broadcast_dict["encrypted_sum"] = (copy.deepcopy(encrypted_chunks), c_upload_list[0]['c_encrypted_params'][1])
        else:
            c_params_list = [c_upload['c_encrypted_params'] for c_upload in c_upload_list]
            encrypted_sum = copy.deepcopy(c_params_list[0])
            for i,c_params in enumerate(c_params_list):
                if i>0:
                    encrypted_sum += copy.deepcopy(c_params)
            self.broadcast_dict["encrypted_sum"] = copy.deepcopy(encrypted_sum)
            
        if self.args.chunk:
            for c_upload in c_upload_list:
                if 'c_encrypted_gen' in c_upload.keys():
                    gen_chunks, gen_chunk_size= c_upload['c_encrypted_gen']
                    self.broadcast_dict["encrypted_gen"] = (copy.deepcopy(gen_chunks), gen_chunk_size)
                    
            # c_gen_list = [c_upload['c_encrypted_gen'][0] for c_upload in c_upload_list]
            # gen_chunks = copy.deepcopy(c_gen_list[0])
            # for i,c_chunks in enumerate(c_params_list):
            #     if i>0:
            #         for chunk_id, chunk in enumerate(c_chunks):
            #             gen_chunks[chunk_id] += copy.deepcopy(c_gen_list[i][chunk_id])
            # self.broadcast_dict["encrypted_gen"] = (copy.deepcopy(gen_chunks), c_upload_list[0]['c_encrypted_gen'][1])
        else:
            for c_upload in c_upload_list:
                if 'c_encrypted_gen' in c_upload.keys():
                    gen_model = c_upload['c_encrypted_gen']
            # c_gen_list = [c_upload['c_encrypted_gen'] for c_upload in c_upload_list]
            # gen_sum = copy.deepcopy(c_gen_list[0])
            # for i,c_params in enumerate(c_params_list):
            #     if i>0:
            #         gen_sum += copy.deepcopy(c_gen_list[i])
                    self.broadcast_dict["encrypted_gen"] = copy.deepcopy(gen_model)   
            
        # if self.args.chunk:
        #     c_params_list = [c_upload['c_encrypted_params'][0] for c_upload in c_upload_list]
        #     c_gen_list = [c_upload['c_encrypted_gen'][0] for c_upload in c_upload_list]
        #     encrypted_chunks = copy.deepcopy(c_params_list[0])
        #     gen_chunks = copy.deepcopy(c_gen_list[0])
        #     for i,c_chunks in enumerate(c_params_list):
        #         if i>0:
        #             for chunk_id, chunk in enumerate(c_chunks):
        #                 encrypted_chunks[chunk_id] += copy.deepcopy(chunk)
        #                 gen_chunks[chunk_id] += copy.deepcopy(c_gen_list[i][chunk_id])
        #     self.broadcast_dict["encrypted_sum"] = (copy.deepcopy(encrypted_chunks), c_upload_list[0]['c_encrypted_params'][1])
        #     self.broadcast_dict["encrypted_gen"] = (copy.deepcopy(gen_chunks), c_upload_list[0]['c_encrypted_gen'][1])
        # else:
        #     c_params_list = [c_upload['c_encrypted_params'] for c_upload in c_upload_list]
        #     c_gen_list = [c_upload['c_encrypted_gen'] for c_upload in c_upload_list]
        #     encrypted_sum = copy.deepcopy(c_params_list[0])
        #     gen_sum = copy.deepcopy(c_gen_list[0])
        #     for i,c_params in enumerate(c_params_list):
        #         if i>0:
        #             encrypted_sum += copy.deepcopy(c_params)
        #             gen_sum += copy.deepcopy(c_gen_list[i])
        #     self.broadcast_dict["encrypted_sum"] = copy.deepcopy(encrypted_sum)
        #     self.broadcast_dict["encrypted_gen"] = copy.deepcopy(gen_sum)   

        return self.broadcast_dict 

