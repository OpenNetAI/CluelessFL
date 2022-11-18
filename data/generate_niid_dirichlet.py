import sys
from tqdm import trange
import numpy as np
import random
import os
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, EMNIST, SVHN
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .categorical_datasets import load_covtype, load_rcv1

random.seed(42)
np.random.seed(42)

def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in trange(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data

def get_dataset(args, mode='train', DATASET='Mnist'):
    global dataset
    
    # set transform
    transform = transforms.Compose(
       [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    cifar10_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    # set data
    if DATASET=='mnist':
        dataset = MNIST(root=args.data_path, train=True if mode=='train' else False, download=True, transform=transform)
        
    elif 'fashionmnist' in DATASET:
        dataset = FashionMNIST(root=args.data_path,train=True if mode=='train' else False, download=True, transform=transform)

    elif DATASET == 'cifar10':
        dataset = CIFAR10(root=args.data_path, train=True if mode=='train' else False, download=True, transform=cifar10_transform)

    elif DATASET == 'emnist':
        dataset = EMNIST(
                args.data_path, 'balanced', train=True if mode=='train' else False, 
                    download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307,), (0.3081,))
                ]))

    elif DATASET == 'svhn':
        dataset = SVHN(
                args.data_path, split=mode, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307,), (0.3081,))
                ]))
    
    elif DATASET == 'covtype':
        dataset = load_covtype(mode=mode, args=args)
        
    elif DATASET == 'rcv1':
        dataset = load_rcv1(mode=mode, args=args)

    else:
        print(f"{DATASET} is not exist. Selectable datasets: mnist, fashionmnist, emnist, avhn, covtype, rcv1")
        sys.exit(0)

    origin_data = dataset.data
    n_sample = len(dataset.data)
    if DATASET == 'svhn':
        SRC_N_CLASS = len(np.unique(dataset.labels))
    else:
        SRC_N_CLASS = len(dataset.classes)
    print(f"n_sample:{n_sample}, SRC_N_CLASS:{SRC_N_CLASS}")
    
    # full batch
    trainloader = DataLoader(dataset, batch_size=n_sample, shuffle=False)

    print("Loading data from storage ...")
    for _, xy in enumerate(trainloader, 0):
        dataset.data, dataset.targets = xy
        
    print("Rearrange data by class...")
    data_by_class = rearrange_data_by_class(
        dataset.data.cpu().detach().numpy(),
        dataset.targets.cpu().detach().numpy(),
        SRC_N_CLASS
    )
    print(f"{mode.upper()} SET:\n  Total #samples: {n_sample}. sample shape: {dataset.data[0].shape}")
    print("  #samples per class:\n", [len(v) for v in data_by_class])
    
    dataset.data = origin_data
    
    return data_by_class, n_sample, SRC_N_CLASS, dataset

def sample_class(SRC_N_CLASS, NUM_LABELS, user_id, label_random=False):
    assert NUM_LABELS <= SRC_N_CLASS
    if label_random:
        source_classes = [n for n in range(SRC_N_CLASS)]
        random.shuffle(source_classes)
        return source_classes[:NUM_LABELS]
    else:
        return [(user_id + j) % SRC_N_CLASS for j in range(NUM_LABELS)]

def devide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample, alpha=0.5, sampling_ratio=0.5):
    min_sample = 10
    min_size = 0 
    ###### Determine Sampling #######
    while min_size < min_sample:
        print("Try to find valid data separation")
        idx_batch=[{} for _ in range(NUM_USERS)]
        samples_per_user = [0 for _ in range(NUM_USERS)]
        max_samples_per_user = sampling_ratio * n_sample / NUM_USERS
        for l in SRC_CLASSES:
            # get indices for all that label
            idx_l = [i for i in range(len(data[l]))]
            np.random.shuffle(idx_l)
            if sampling_ratio < 1:
                samples_for_l = int( min(max_samples_per_user, int(sampling_ratio * len(data[l]))) )
                idx_l = idx_l[:samples_for_l]
                print(l, len(data[l]), len(idx_l))
            # dirichlet sampling from this label
            proportions=np.random.dirichlet(np.repeat(alpha, NUM_USERS))
            # re-balance proportions
            proportions=np.array([p * (n_per_user < max_samples_per_user) for p, n_per_user in zip(proportions, samples_per_user)])
            proportions=proportions / proportions.sum()
            proportions=(np.cumsum(proportions) * len(idx_l)).astype(int)[:-1]
            # participate data of that label
            for u, new_idx in enumerate(np.split(idx_l, proportions)):
                # add new idex to the user
                idx_batch[u][l] = new_idx.tolist()
                samples_per_user[u] += len(idx_batch[u][l])
        min_size=min(samples_per_user)

    ###### CREATE USER DATA SPLIT #######
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels=[set() for _ in range(NUM_USERS)]
    print("processing users...")
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if len(indices) == 0: continue
            X[u] += data[l][indices].tolist()
            y[u] += (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)

    return X, y, Labels, idx_batch, samples_per_user

def divide_test_data(NUM_USERS, SRC_CLASSES, test_data, Labels, unknown_test):
    # Create TEST data for each user.
    test_X = [[] for _ in range(NUM_USERS)]
    test_y = [[] for _ in range(NUM_USERS)]
    idx = {l: 0 for l in SRC_CLASSES}
    for user in trange(NUM_USERS):
        if unknown_test: # use all available labels
            user_sampled_labels = SRC_CLASSES
        else:
            user_sampled_labels =  list(Labels[user])
        for l in user_sampled_labels:
            num_samples = int(len(test_data[l]) / NUM_USERS )
            assert num_samples + idx[l] <= len(test_data[l])
            test_X[user] += test_data[l][idx[l]:idx[l] + num_samples].tolist()
            test_y[user] += (l * np.ones(num_samples)).tolist()
            assert len(test_X[user]) == len(test_y[user]), f"{len(test_X[user])} == {len(test_y[user])}"
            idx[l] += num_samples
    return test_X, test_y

def Generate_niid_dirichelet(args):
    print("Using dirichlet to divide data......")
    print("Number of clinets: {}".format(args.num_clients))
    print("Min # of samples per clients: {}".format(args.min_sample))
    print("Alpha for Dirichlet Distribution: {}".format(args.alpha))
    print("Ratio for Sampling Training Data: {}".format(args.sampling_ratio))
    NUM_USERS = args.num_clients

    def process_user_data(mode, data, n_sample, SRC_CLASSES, Labels=None, unknown_test=0, DATASET="Mnist"):
        if mode == 'train':
            X, y, Labels, idx_batch, samples_per_user  = devide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS, args.min_sample, args.alpha, args.sampling_ratio)
        if mode == 'test':
            assert Labels != None or unknown_test
            X, y = divide_test_data(NUM_USERS, SRC_CLASSES, data, Labels, unknown_test)
        dataset={'users': [], 'user_data': {}, 'num_samples': []}
        for i in range(NUM_USERS):
            uname='{0:05d}'.format(i)
            dataset['users'].append(uname)
            dataset['user_data'][uname]={
                    'x': torch.tensor(X[i], dtype=torch.float32),
                    'y': torch.tensor(y[i], dtype=torch.int64)}
            dataset['num_samples'].append(len(X[i]))
        
        if mode == 'train':
            for u in range(NUM_USERS):
                print("{} samples in total".format(samples_per_user[u]))
                train_info = ''
                # train_idx_batch, train_samples_per_user
                n_samples_for_u = 0
                for l in sorted(list(Labels[u])):
                    n_samples_for_l = len(idx_batch[u][l])
                    n_samples_for_u += n_samples_for_l
                    train_info += "c={},n={}| ".format(l, n_samples_for_l)
                print(train_info)
                print("{} Labels/ {} Number of training samples for user [{}]:".format(len(Labels[u]), n_samples_for_u, u))
            return dataset, Labels, idx_batch, samples_per_user
        else:
            return dataset

    print("Reading source dataset:{}".format(args.dataset))

    train_data, n_train_sample, SRC_N_CLASS, origin_train = get_dataset(args, mode='train', DATASET=args.dataset)
    test_data, n_test_sample, SRC_N_CLASS, origin_test = get_dataset(args, mode='test', DATASET=args.dataset)
    SRC_CLASSES=[l for l in range(SRC_N_CLASS)]
    random.shuffle(SRC_CLASSES)
    print("{} labels in total.".format(len(SRC_CLASSES)))
    traindata, Labels, idx_batch, samples_per_user = process_user_data('train', train_data, n_train_sample, SRC_CLASSES,
                                                            DATASET=args.dataset)
    testdata = process_user_data('test', test_data, n_test_sample, SRC_CLASSES,
                      Labels=Labels, unknown_test=args.unknown_test, DATASET=args.dataset)
    print("Finish Generating User samples")

    return traindata, testdata, origin_test, SRC_CLASSES

