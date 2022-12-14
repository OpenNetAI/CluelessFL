import argparse
import numpy
import torch
from FLAlgorithms.FedAvg_Algorithm import FedAvg
from FLAlgorithms.FedDGT_Algorithm import FedDGT
from FLAlgorithms.MOON_Algorithm import MOON
from FLAlgorithms.FedProx_Algorithm import FedProx
from FLAlgorithms.FedScaf_Algorithm import FedScaf
from FLAlgorithms.FedDC_Algorithm import FedDC
import random


def make_print_to_file(args):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime
    path = args.stdout
    if not os.path.isdir(path):
        os.mkdir(path, mode=0o755)
 
    class Logger(object):
        def __init__(self, filename="pretrained_public_mnist_initial.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
 
    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d'+'_'+args.dataset+'_'+args.model+'_'+args.algorithm)
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))


def main(args):
    if args.device!="cuda":
        exit("Error: only CUDA is supported for now!")
    torch.cuda.set_device(args.gpu)
    random.seed(12345)
    numpy.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345) 

    for run_time in range(args.times):
        # Initialize algorithm
        if args.algorithm == 'feddgt': 
            algorithm = FedDGT(args, run_time)
        elif args.algorithm == 'fedavg': 
            algorithm = FedAvg(args, run_time)
        elif args.algorithm == 'fedprox': 
            algorithm = FedProx(args, run_time) 
        elif args.algorithm == 'fedscaf': 
            algorithm = FedScaf(args, run_time) 
        elif args.algorithm == 'moon': 
            algorithm = MOON(args, run_time) 
        elif args.algorithm == 'feddc': 
            algorithm = FedDC(args, run_time) 
        else:
            exit("Error: Unsupported algorithm!")
        algorithm.run_job(args, run_time)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data-path", type=str, default="data/mnist")
    parser.add_argument("--model", type=str, default="lenet")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gen-batch-size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--num-glob-iters", type=int, default=300)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=10, help="Number of the total clients")
    parser.add_argument("--selected-num", type=int, default=10, help="Number of the selected clients")
    parser.add_argument("--K", type=int, default=20, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--device", type=str, default="cuda", help="Only CUDA is supported for now")
    parser.add_argument("--board-dir", type=str, default="board/test", help="directory path to save tensorboard")
    parser.add_argument("--stdout", type=str, default="stdout/test", help="directory path to save logs")
    parser.add_argument("--chunk", action='store_true', help="communicate chunks")
    parser.add_argument("--dirichlet", type=int, default=1, help="whether to use dirichlet division to process dataset")
    parser.add_argument("--min-sample", type=int, default=10, help="Min number of samples per user.")
    parser.add_argument("--sampling-ratio", type=float, default=1, help="Ratio for sampling training samples.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="alpha in Dirichelt distribution (smaller means larger heterogeneity)")
    parser.add_argument("--training-gen-batch",  type=int, default=20)
    parser.add_argument("--training-gen-step",  type=int, default=5)
    parser.add_argument("--warm-rounds", type=int, default=0)
    args = parser.parse_args()
    make_print_to_file(args=args)

    print("=" * 80)
    print(args)
    print("=" * 80)
    main(args)
