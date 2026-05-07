import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, Muon

from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parents[2]))

from src.data import get_cifar10, randomize_label
from src.model import DNN
from src.train import train, evaluate
from src.metric import compute_layerwise_metrics
from src.utils import write_json, extract_feature


DATA_DICT = {
    'cifar10': (3*32*32, 10, get_cifar10)
    }


if __name__ == "__main__":
    num_epoch = 200
    hidden_dim = 256
    num_hidden_layer = 3
    lr = 0.005
    save_path = Path(__file__).parents[2] / 'result' / 'random_label'
    os.makedirs(save_path, exist_ok=True)
    for data in DATA_DICT.keys():
        data_info = DATA_DICT[data]
        for p in [0.25, 0.75, 0.9]:
            print(f'######### Running {data} Probability {p}#########')
            trainset, testset = data_info[2]()
            trainset, testset = randomize_label(trainset, testset, p=p, c=10)
            torch.manual_seed(0)
            train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
            test_loader = DataLoader(testset, batch_size=64, shuffle=False)

            ret_dict = dict()
            model = DNN(input_dim=data_info[0], 
                        hidden_dim=hidden_dim, 
                        output_dim=data_info[1], 
                        num_hidden=num_hidden_layer,
                        bias=True).to('cuda')
            optimizer = SGD(model.parameters(), lr=lr, momentum=0.95)
            loss_fn = CrossEntropyLoss()
            
            train_loss_list, train_accuracy_list = list(), list()
            test_loss_list, test_accuracy_list = list(), list()
            train_surrogate_list, test_surrogate_list = list(), list()
            train_target_linearity_list, test_target_linearity_list = list(), list()

            train_inputs, train_features, train_targets = extract_feature(model, train_loader)
            test_inputs, test_features, test_targets = extract_feature(model, test_loader)
            train_surrogate, train_target_linearity = compute_layerwise_metrics(model, train_inputs, train_features, train_targets)
            test_surrogate, test_target_linearity = compute_layerwise_metrics(model, test_inputs, test_features, test_targets)
            train_surrogate_list.append(train_surrogate)
            test_surrogate_list.append(test_surrogate)
            train_target_linearity_list.append(train_target_linearity)
            test_target_linearity_list.append(test_target_linearity)

            for epoch in range(num_epoch):
                train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, epoch+1)
                test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, epoch+1)
                
                train_inputs, train_features, train_targets = extract_feature(model, train_loader)
                test_inputs, test_features, test_targets = extract_feature(model, test_loader)
                train_surrogate, train_target_linearity = compute_layerwise_metrics(model, train_inputs, train_features, train_targets)
                test_surrogate, test_target_linearity = compute_layerwise_metrics(model, test_inputs, test_features, test_targets)
                train_surrogate_list.append(train_surrogate)
                test_surrogate_list.append(test_surrogate)
                train_target_linearity_list.append(train_target_linearity)
                test_target_linearity_list.append(test_target_linearity)

                train_loss_list.append(round(train_loss, 4))
                train_accuracy_list.append(round(train_accuracy, 4))
                test_loss_list.append(round(test_loss, 4))
                test_accuracy_list.append(round(test_accuracy, 4))
                ret_dict = {
                    'train_accuracy': train_accuracy_list,
                    'train_loss': train_loss_list,
                    'test_accuracy': test_accuracy_list,
                    'test_loss': test_loss_list,
                    'train_surrogate': train_surrogate_list,
                    'test_surrogate': test_surrogate_list,
                    'train_target_linearity': train_target_linearity_list,
                    'test_target_linearity': test_target_linearity_list
                }
                write_json(ret_dict, save_path / f'{data}_{p}.json')
