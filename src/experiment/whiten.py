import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, Muon

from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parents[2]))

from src.data import get_cifar100, get_cifar10, get_mnist, get_svhn
from src.model import DNN
from src.train import train, evaluate
from src.optim import GramFace, whitening
from src.utils import write_json


DATA_DICT = {
    'mnist': (28*28, 10, get_mnist),
    'svhn': (3*32*32, 10, get_svhn),
    'cifar10': (3*32*32, 10, get_cifar10), 
    'cifar100': (3*32*32, 100, get_cifar100) # Use momentum 0.8 for CIFAR 100 and 0.95 for the others
    }


if __name__ == "__main__":
    num_iter = 5
    num_epoch = 20
    hidden_dim = 256
    num_hidden_layer = 2
    lr_list = [0.01, 0.005, 0.1, 0.05]
    for base_optim in [SGD]:
        for transform in [whitening]:
            optim_name = base_optim.__name__.lower()
            transform_name = 'orig' if transform is None else 'whiten'
            save_path = Path(__file__).parents[2] / 'result' / 'whitening'
            os.makedirs(save_path, exist_ok=True)
            for data in DATA_DICT.keys():
                print(f'######### Running {data} #########')
                data_info = DATA_DICT[data]
                trainset, testset = data_info[2]()
                ret_dict = dict()
                for lr in lr_list:
                    lr_dict = dict()
                    for i in range(num_iter):
                        print(f'[Running] Optim: {optim_name} ({transform_name}), Learning rate: {lr}, iteration: {i+1} / {num_iter}')
                        torch.manual_seed(i)
                        train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
                        test_loader = DataLoader(testset, batch_size=64, shuffle=False)

                        model = DNN(input_dim=data_info[0], 
                                    hidden_dim=hidden_dim, 
                                    output_dim=data_info[1], 
                                    num_hidden=num_hidden_layer,
                                    bias=True).to('cuda')
                        optimizer = GramFace(base_optim, 
                                            model.parameters(), 
                                            transform_fn=transform, 
                                            requires_2d=False, 
                                            lr=lr,
                                            momentum=0.85)
                        loss_fn = CrossEntropyLoss()

                        loss_list, accuracy_list = list(), list()
                        for epoch in range(num_epoch):
                            train(model, train_loader, loss_fn, optimizer, epoch+1)
                            loss, accuracy = evaluate(model, test_loader, loss_fn, epoch+1)
                            loss_list.append(round(loss, 4))
                            accuracy_list.append(round(accuracy, 4))
                        lr_dict[i] = {
                            'accuracy': accuracy_list,
                            'loss': loss_list
                        }
                    ret_dict[lr] = lr_dict
                write_json(ret_dict, save_path / f'{data}_{transform_name}.json')
