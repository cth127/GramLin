import torch
from tqdm import tqdm
import json
import numpy as np


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(res, path) :
    with open(path, 'w', encoding = 'UTF-8') as f:
        json.dump(res, f, indent = 4)
        f.close()


def extract_feature(model, dataloader, n_sampling=None, device='cuda'):
    """
    Extract features from the model for the given dataloader.
    
    Args:
        model: The neural network model.
        dataloader: DataLoader providing the data.
        device: Device to perform computation on (CPU or GPU).
    
    Returns:
        features: Extracted features from the model.
        targets: Corresponding targets for the features.
    """
    model.eval()
    input_features = []
    features = []
    targets = []

    if isinstance(n_sampling, int):
        sampling_index = np.random.choice(np.arange(len(dataloader.dataset)), size=n_sampling, replace=False)
    else:
        sampling_index = np.arange(len(dataloader.dataset))

    with torch.no_grad():
        for inputs, target in tqdm(dataloader, desc="[Extracting Features]"):
            inputs = inputs.to(device)
            _, hiddens = model(inputs, return_hidden=True)
            input_features.append(inputs.view(inputs.shape[0], -1))
            features.append(hiddens)
            targets.append(target.to(device))
    features_ = [[] for _ in range(len(model.layers)-1)]
    for i in features:
        for j in range(len(model.layers)-1):
            features_[j].append(i[j])
    input_features = torch.cat(input_features)[sampling_index]
    features = [torch.cat(i)[sampling_index] for i in features_]
    targets = torch.cat(targets)[sampling_index]
    return input_features, features, targets
    

def newtonschulz(A: torch.Tensor, steps: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes UV^T of A = USV^T via Newton-Schulz iteration.
    """
    assert A.dim() == 2

    X = A / (A.norm() + eps)
    
    for _ in range(steps):
        XtX = X.T @ X
        # Quintic update — degree 5 polynomial, cubic convergence rate but larger basin
        X = X @ (15 * torch.eye(XtX.shape[0], device=X.device, dtype=X.dtype)
                 - 10 * XtX
                 +  3 * XtX @ XtX) / 8

    return X

