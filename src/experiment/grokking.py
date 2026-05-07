import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, ReLU
from torch.optim import SGD, AdamW
from torch.nn.functional import one_hot

from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parents[2]))

from src.data import CustomDataset
from src.grok import operation_mod_p_data
from src.model import DNN
from src.train import train, evaluate
from src.metric import compute_layerwise_metrics, compute_layerwise_regression
from src.utils import write_json, extract_feature


if __name__ == "__main__":
    num_epoch = 500
    hidden_dim = 256
    num_hidden_layer = 3
    lr = 0.001
    save_path = Path(__file__).parents[2] / 'result' / 'grokking' / 'weight_decay'
    os.makedirs(save_path, exist_ok=True)
    
    for wd in [0.0, 0.25, 0.5, 0.75, 1.0]:
        X, y = operation_mod_p_data('x+y', 61)
        X = one_hot(X).view(-1, 61 * 2).to(torch.double)
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
        trainset = CustomDataset(train_X, train_y)
        testset = CustomDataset(test_X, test_y)
        
        torch.manual_seed(0)
        train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
        test_loader = DataLoader(testset, batch_size=32, shuffle=False)
        train_steps = len(train_loader)

        ret_dict = dict()
        model = DNN(input_dim=61 * 2,
                    hidden_dim=hidden_dim, 
                    output_dim=61,
                    num_hidden=num_hidden_layer,
                    bias=True).to('cuda')

        model.nonlin = ReLU()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
        loss_fn = CrossEntropyLoss()
        
        train_loss_list, train_accuracy_list = list(), list()
        test_loss_list, test_accuracy_list = list(), list()
        train_surrogate_list, test_surrogate_list = list(), list()
        train_target_linearity_list, test_target_linearity_list = list(), list()
        kernel_regression_list = list()

        train_inputs, train_features, train_targets = extract_feature(model, train_loader)
        test_inputs, test_features, test_targets = extract_feature(model, test_loader)
        train_surrogate, train_target_linearity = compute_layerwise_metrics(model, train_inputs, train_features, train_targets)
        test_surrogate, test_target_linearity = compute_layerwise_metrics(model, test_inputs, test_features, test_targets)
        kernel_regression = compute_layerwise_regression(train_features, train_targets, test_features, test_targets)

        train_surrogate_list.append(train_surrogate)
        test_surrogate_list.append(test_surrogate)
        train_target_linearity_list.append(train_target_linearity)
        test_target_linearity_list.append(test_target_linearity)
        kernel_regression_list.append(kernel_regression)

        for epoch in range(num_epoch):
            train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, epoch+1)
            test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, epoch+1)
            
            train_inputs, train_features, train_targets = extract_feature(model, train_loader)
            test_inputs, test_features, test_targets = extract_feature(model, test_loader)
            train_surrogate, train_target_linearity = compute_layerwise_metrics(model, train_inputs, train_features, train_targets)
            test_surrogate, test_target_linearity = compute_layerwise_metrics(model, test_inputs, test_features, test_targets)
            kernel_regression = compute_layerwise_regression(train_features, train_targets, test_features, test_targets)

            train_surrogate_list.append(train_surrogate)
            test_surrogate_list.append(test_surrogate)
            train_target_linearity_list.append(train_target_linearity)
            test_target_linearity_list.append(test_target_linearity)
            kernel_regression_list.append(kernel_regression)

            train_loss_list.append(round(train_loss, 4))
            train_accuracy_list.append(round(train_accuracy, 4))
            test_loss_list.append(round(test_loss, 4))
            test_accuracy_list.append(round(test_accuracy, 4))

        ret_dict = {
            'steps': train_steps,
            'train_accuracy': train_accuracy_list,
            'train_loss': train_loss_list,
            'test_accuracy': test_accuracy_list,
            'test_loss': test_loss_list,
            'train_surrogate': train_surrogate_list,
            'test_surrogate': test_surrogate_list,
            'train_target_linearity': train_target_linearity_list,
            'test_target_linearity': test_target_linearity_list,
            'kernel_regression_error': kernel_regression_list
        }
        write_json(ret_dict, save_path / f'grokking_{int(wd * 100)}.json')
