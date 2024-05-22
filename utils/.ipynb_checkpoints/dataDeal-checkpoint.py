import torch
import torchvision
from torchvision.datasets import FashionMNIST, Omniglot, MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn import svm
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

# NB: background=True selects the train set, background=False selects the test set
# It's the nomenclature from the original paper, we just have to deal with it
def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices

def sample_data(args):
    dataset = globals()[args.data](
        root=args.filepath+args.data,
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=args.N_CHANNEL),
                transforms.Resize([args.N_SIZE, args.N_SIZE]),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )
    
    if args.data.lower() == 'mnist':
        from torchvision.datasets import MNIST as Dataset
    elif args.data.lower() == 'fashionmnist':
        from torchvision.datasets import FashionMNIST as Dataset
        
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(args.N_SIZE), torchvision.transforms.ToTensor()])
    dataset, dataloader = {}, {}
    images, labels = {'train':[], 'query':[], 'test':[]}, {'train':[], 'query':[], 'test':[]}
    dataset['train'] = Dataset('../data/', train = True, download = True, transform = transform)
    dataset['test'] = Dataset('../data/', train = False, download = True, transform = transform)
    
    for ty in ['train', 'query','test']:
        
        for cate in args.classes:
            idxs = []
            if ty == 'query':
                idxs += get_indices(dataset['train'], int(cate))
            else:
                idxs += get_indices(dataset[ty], int(cate))
                
            sampler = SubsetRandomSampler(idxs)
            
            if ty == 'train':
                dataloader[ty] = DataLoader(dataset[ty], batch_size=args.N_SHOT, drop_last = True, sampler = sampler)
            elif ty == 'query':
                dataloader[ty] = DataLoader(dataset['train'], batch_size=args.N_QUERY, drop_last = True, sampler = sampler)
            elif ty == 'test':
                dataloader[ty] = DataLoader(dataset[ty], batch_size=args.N_TEST, drop_last = True, sampler = sampler)
        
        
            images_sample, labels_sample =  next(iter(dataloader[ty]))
            images[ty] += images_sample
            labels[ty] += labels_sample
    return images, labels

def sample_data_qac(args):
    dataset = globals()[args.data](
        root=args.filepath+args.data,
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=args.N_CHANNEL),
                transforms.Resize([args.N_SIZE, args.N_SIZE]),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )
    
    if args.data.lower() == 'mnist':
        from torchvision.datasets import MNIST as Dataset
    elif args.data.lower() == 'fashionmnist':
        from torchvision.datasets import FashionMNIST as Dataset
        
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(args.N_SIZE), torchvision.transforms.ToTensor()])
    dataset, dataloader = {}, {}
    images, labels = {'train':[], 'validate':[],'test':[]}, {'train':[], 'validate':[], 'test':[]}
    dataset['train'] = Dataset('../data/', train = True, download = True, transform = transform)
    dataset['test'] = Dataset('../data/', train = False, download = True, transform = transform)
    
    for ty in ['train', 'validate', 'test']:
        
        for cate in args.classes:
            idxs = []
            if ty == 'validate':
                idxs += get_indices(dataset['test'], int(cate))
            else:
                idxs += get_indices(dataset[ty], int(cate))
            
            sampler = SubsetRandomSampler(idxs)
            
            if ty == 'train':
                dataloader[ty] = DataLoader(dataset[ty], batch_size=args.N_TRAIN, drop_last = True, sampler = sampler)
            elif ty == 'validate':
                dataloader[ty] = DataLoader(dataset['test'], batch_size=args.N_VALIDATE, drop_last = True, sampler = sampler)
            elif ty == 'test':
                dataloader[ty] = DataLoader(dataset[ty], batch_size=args.N_TEST, drop_last = True, sampler = sampler)
        
            images_sample, labels_sample =  next(iter(dataloader[ty]))
            images[ty] += images_sample
            labels[ty] += labels_sample
    
    return images, labels

def loadPKL(args):
    import pickle

    # 指定保存文件的路径（与保存时相同）
    file_path = args.filepath+args.data+'/'+args.data+'_shot='+str(args.N_SHOT)+".pkl" #"/data/hzh-2022/pennylane/data/FMNIST.pkl"

    # 使用pickle加载变量，并通过关键词访问它们
    with open(file_path, 'rb') as file:
        loaded_variables = pickle.load(file)

    # 现在，你可以通过关键词来访问加载的变量
    example_support_images = loaded_variables['example_support_images']
    example_support_labels = loaded_variables['example_support_labels']
    example_query_images = loaded_variables['example_query_images']
    example_query_labels = loaded_variables['example_query_labels']
    example_class_ids = loaded_variables['example_class_ids']
    
    return [example_support_images, example_support_labels, example_query_images, example_query_labels, example_class_ids]
    