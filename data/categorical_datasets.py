from torch.utils.data import Dataset
import numpy as np
import torch

class CategoricalDataset(Dataset):
    def __init__(self, features, labels, model='cmlp'):
        self.data = torch.tensor(features)
        self.labels = torch.tensor(labels)
        self.classes = ['0 - 0', '1 - 1']
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def load_covtype(mode = 'train', model='cmlp', args=None):
    if mode == 'train':
        features = np.load(args.data_path + 'X_train_scale.npy')
        labels = np.load(args.data_path + 'y_train_scale.npy')
    else:
        features = np.load(args.data_path + 'X_test_scale.npy') 
        labels = np.load(args.data_path + 'y_test_scale.npy')
    
    dataset = CategoricalDataset(features, labels, 'cmlp')
    
    print("Total samples: ", len(dataset))
    return dataset

def load_rcv1(mode = 'train', model='cmlp', args=None):
    if mode == 'train':
        features = np.load(args.data_path + 'X_train.npy')
        labels = np.load(args.data_path + 'y_train.npy')
    else:
        features = np.load(args.data_path + 'X_test.npy') 
        labels = np.load(args.data_path + 'y_test.npy')
    
    dataset = CategoricalDataset(features, labels, 'cmlp')
    
    print("Total samples: ", len(dataset))
    return dataset