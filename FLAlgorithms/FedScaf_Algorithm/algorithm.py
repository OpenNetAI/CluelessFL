
from .server import ServerFedScaf
from .clients import generate_clients, generate_niid_dirichelet_Clients
from model_utils.create_model import create_model, create_mlp_model
import torch
import copy
import os
from tensorboardX import SummaryWriter

class FedScaf(object):
    def __init__(self, args, run_time):
        if args.dataset == 'emnist':
            model = create_model(args.model, num_classes=47)
        elif args.dataset in ('covtype', 'rcv1'):
            input_size = {
                'covtype': 54,
                'rcv1': 47236
            }[args.dataset]
            model = create_mlp_model('cmlp', input_size, [32, 16, 8], 2, 2)
        else:
            model = create_model(args.model)
        self.Server = ServerFedScaf(args, model, run_time)
        if args.dirichlet == 0:
            self.testset, self.Clients = generate_clients(args, model)
        else:
            self.testset, self.Clients = generate_niid_dirichelet_Clients(args, model)
        self.writer = SummaryWriter(os.path.join(args.board_dir, args.algorithm))
        self.max_acc = 0
    
    def run_job(self, args, run_time):
        torch.manual_seed(run_time)
        print("\n\n         [ Start running time {} ]           \n\n".format(run_time))
        
        testloader = self.get_testloader(args, self.testset)
        # run job
        for epoch in range(args.num_glob_iters):
            # select clients to participate
            selected_ids = self.Server.select_clients(epoch, args.selected_num)
            selected_clients = [self.Clients[i] for i in selected_ids]

            # local training in the clients
            c_upload_list = []
            for id, c in enumerate(selected_clients):
                c_upload = c.train(args, epoch, self.Server.broadcast_dict)
                c_upload_list.append(c_upload)
            
            # aggregate in the PS
            self.Server.aggregate(c_upload_list)

            # test global model accuracy of round epoch
            self.test(args, epoch, testloader)
            
        print('Max accuracy:', self.max_acc)

    def get_testloader(self, args, dataset_test):
        # prepare dataset
        if args.dataset=='mnist':
            testloader = torch.utils.data.DataLoader(dataset_test,num_workers=2,batch_size=100,shuffle=False)
        elif args.dataset=='fashionmnist':
            testloader = torch.utils.data.DataLoader(dataset_test,num_workers=2,batch_size=100,shuffle=False)
        elif args.dataset=='cifar10':
            testloader = torch.utils.data.DataLoader(dataset_test,num_workers=2,batch_size=100,shuffle=False)
        elif args.dataset=='emnist':
            testloader = torch.utils.data.DataLoader(dataset_test,num_workers=2, batch_size=100, shuffle=False)
        elif args.dataset=='svhn':
            testloader = torch.utils.data.DataLoader(dataset_test,num_workers=2, batch_size=100, shuffle=False)
        elif args.dataset in ('covtype', 'rcv1'):
            testloader = torch.utils.data.DataLoader(dataset_test,num_workers=2, batch_size=64, shuffle=False)
        return testloader

    def test(self, args, epoch, testloader):
        # get test accuracy
        if args.partial_sharing:
            testmodel = copy.deepcopy(self.Clients[0].model)
        else:
            testmodel = copy.deepcopy(self.Server.model)
        testmodel.eval()
        with torch.no_grad():
            correct = 0
            total = 0 
            for i, data in enumerate(testloader):
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                _,outputs = testmodel(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        del testmodel
        print('Round [{}/{}]: {:.2f} ({} / {})'.format(epoch, args.num_glob_iters, (100 * float(correct) / total), correct, total))
        if (100 * float(correct) / total) > self.max_acc:
            self.max_acc = 100 * float(correct) / total
            print('max accuracy achieved: {}'.format(self.max_acc))
        self.writer.add_scalar(f'test_acc', (100 * float(correct) / total), epoch)
        

