import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, Muon

from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parents[3]))

from src.data import get_cifar100, get_cifar10, get_svhn
from src.model import CNN
from src.train import train, evaluate
from src.optim import GramFace, whitening
from src.utils import write_json


DATA_DICT = {
    'svhn': ((32, 32, 3), 10, get_svhn),
    'cifar10': ((32, 32, 3), 10, get_cifar10),
    'cifar100': ((32, 32, 3), 100, get_cifar100)
    }


if __name__ == "__main__":
    num_iter = 5
    num_epoch = 20
    lr_list = [0.1, 0.05, 0.01, 0.005]
    for base_optim in [SGD]:
        for transform in [whitening]:
            optim_name = base_optim.__name__.lower()
            transform_name = 'orig' if transform is None else 'whiten'
            save_path = Path(__file__).parents[2] / 'result' / 'whitening' / 'cnn'
            os.makedirs(save_path, exist_ok=True)
            for data in DATA_DICT.keys():
                print(f'######### Running {data} - {str(transform)} #########')
                data_info = DATA_DICT[data]
                trainset, testset = data_info[2]()
                ret_dict = dict()
                for lr in lr_list:
                    lr_dict = dict()
                    for i in range(num_iter):
                        print(f'[Running] learning rate: {lr}, iteration: {i+1} / {num_iter}')
                        torch.manual_seed(i)
                        train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
                        test_loader = DataLoader(testset, batch_size=64, shuffle=False)

                        model = CNN(input_dim=(data_info[0]),
                                    num_conv=3,
                                    num_maxpool=2,
                                    maxpool_layer=(0,2),
                                    out_channels=16,
                                    output_dim=data_info[1],
                                    bias=True).to('cuda')
                        if optim_name == 'muon' or transform_name == 'whiten':
                            requires_2d = True
                        else:
                            requires_2d = False
                        optimizer = GramFace(base_optim, 
                                            model.parameters(), 
                                            transform_fn=transform, 
                                            requires_2d=True, 
                                            lr=lr,
                                            momentum=0.95)
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
