import torch
from torchvision.datasets import FashionMNIST, Omniglot, MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

import random

from sklearn import svm
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

import sys
sys.path.append('/your_path/QIAL')
sys.path.append('/your_path/QIAL/utils')
from dataDeal import sample_data, sample_data_qac, loadPKL
from methods import *
from metrics import *

import argparse

def set_params():
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--model', type=str, default='QFIC', help='Model Type')
    parser.add_argument('--strategy', type=str, default='QUANTUM', help='Strategy Type: RAND/ENTRO/QUANTUM/ALL')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')

    parser.add_argument('--filepath', type=str, default='/your_path/QAC/data/', help='File path of image dataset')
    parser.add_argument('--data', type=str, default='MNIST', help='The dataset: FashionMNIST/MNIST/MNIST01')
    parser.add_argument('--data_method', type=str, default='SAMPLE', help='The data method: SAMPLE/LOADPKL/LOADPKL_SAMPLE')

    parser.add_argument('--N_SIZE', type=int, default=16, help='The size of image in few-shot learning')
    parser.add_argument('--N_CHANNEL', type=int, default=1, help='The number of channels in few-shot learning each way')
    parser.add_argument('--N_WAY', type=int, default=2, help='The number of ways in few-shot learning')

    parser.add_argument('--N_TRAIN', type=int, default=200, help='The number of available train samples in few-shot learning each way')
    parser.add_argument('--N_VALIDATE', type=int, default=10, help='The number of validate samples in few-shot learning each way')
    parser.add_argument('--N_TEST', type=int, default=200, help='The number of test samples in few-shot learning')
    
    parser.add_argument('--N_SHOT', type=int, default=10, help='The number of labeled shots in few-shot learning each way')


    parser.add_argument('--N_ACTIVE', type=int, default=0, help='The number of active learning samples in few-shot learning')
    parser.add_argument('--N_ACTIVE_TIMES', type=int, default=0, help='The number of active learning times in few-shot learning')
    

    parser.add_argument('--classes', nargs=4, type=int, default=[0, 1, 2, 3], help='The ids of four classes')
    parser.add_argument('--num_qubits', type=int, default=9, help='Num of qubit')
    parser.add_argument('--num_layers', type=int, default=4, help='The number of layers of QFIC')
    parser.add_argument('--cl_loss_weight', type=float, default=0.1, help='The weight of contrastive loss')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='The learning rate of QFIC')
    
    return parser


def setup_seed(args):
    if args.seed!=-1:
         torch.manual_seed(args.seed)
         torch.cuda.manual_seed_all(args.seed)
         np.random.seed(args.seed)
         random.seed(args.seed)
         torch.backends.cudnn.deterministic = True
    else:
        args.seed = random.randint(0, 10000)
        print('seed='+str(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True 

def generate_task(args):
    if args.data_method=='SAMPLE':
        [example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,] = sample_data(args)
    elif args.data_method=='LOADPKL':
            [example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,] = loadPKL(args)
    
    return example_support_images, example_support_labels, example_query_images, example_query_labels, example_class_ids