# **CluelessFL: A Server-Clueless Federated Learning Platform with Knowledge Distillation**

Prototype of our server-clueless FL platform, CluelessFL. The paper is under review at SIGMOD2023.

This branch implements HE compared to the master branch.

## Datasets

- MNIST
- FashionMNIST
- EMNIST
- SVHN
- CIFAR10
- Covtype
- RCV1

## Algorithms

- FedAvg
- FedProx
- SCAFFOLD
- MOON
- FedDC
- FedDGT (the proposed algorithm)

## Run experiments
Run MNIST example:

```
python main.py --dataset mnist --model lenet --alpha 0.1 --board-dir board/test --stdout stdout/test
```
Look at learning curves in TensorBoard:
```
tensorboard --logdir board/test
```