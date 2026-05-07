import torch

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_layerwise_metrics(model, features, targets, centered=True, categorize=True, pca_dim=256):
    if categorize:
        targets = torch.nn.functional.one_hot(targets).to(features[0].dtype)
    else:
        targets = targets.view(-1, 1)
    if centered:
        targets = targets - targets.mean(dim=0).unsqueeze(0)
    features = [i.to('cpu') for i in features]

    surrogate_ret, norms_ret = list(), list()
    for n, feature in enumerate(features):
        X = feature.to('cuda')
        W = model.layers[n+1].weight.data.clone()
        if len(X.shape) == 4 and len(W.shape) == 4:
            X = model.unfold(X)
            W = model.unfold(W).squeeze(2)
        else:
            X = X.view(X.shape[0], -1)
        surrogate_, norm  = surrogate(targets, X, W, pca_dim)
        surrogate_ret.append(surrogate_)
        norms_ret.append(norm)

    target_linearity_ret = list()
    for feature in features:
        X = feature.to('cuda')
        if len(X.shape) == 4:
            X = model.unfold(X)
            X = X.view(X.shape[0], -1)
        if X.shape[-1] > 1000:
            _, _, V = torch.pca_lowrank(X, q=pca_dim)
            X = torch.matmul(X, V)
        target_linearity_ = target_linearity(targets, X)
        target_linearity_ret.append(target_linearity_)
    return surrogate_ret, target_linearity_ret, norms_ret


def target_linearity(Y, H, eps=None):
    with torch.no_grad():
        G = H.T @ H
        if eps is None:
            try:
                Ginv = torch.linalg.inv(G)
            except:
                G = G + 1e-5 * torch.eye(H.shape[1]).to(G.device)
                Ginv = torch.linalg.inv(G)
        else:
            G = G + eps * torch.eye(H.shape[1]).to(G.device)
            Ginv = torch.linalg.inv(G)
        Yhat = torch.linalg.multi_dot((H, Ginv, H.T, Y))
        pred_error = (Y - Yhat).norm(p='fro') ** 2
        if Y.shape[1] == 1:
            Y_centered = Y - Y.mean(dim=0)
        else:
            Y_centered = Y - Y.mean(dim=1).view(Y.shape[0], -1)
        mean_error = Y_centered.norm(p='fro') ** 2
        ret = 1 - (pred_error / mean_error)
        return ret.item()


def surrogate(Y, X, W, pca_dim):
    with torch.no_grad():
        if len(X.shape) == 3:
            Z = X.transpose(1, 2) @ W.T
            Z = Z.view(Z.shape[0], -1)
        else:
            Z = X @ W.T
        
        if Z.shape[-1] > 1000:
            _, _, V = torch.pca_lowrank(Z, q=pca_dim)
            Z = torch.matmul(Z, V)

        G = Z @ Z.T
        ret = torch.linalg.multi_dot((Y.T, G, Y)).diag()
        norm = Z.norm(p='fro').item() ** 2
        return ret.mean().item(), norm
    