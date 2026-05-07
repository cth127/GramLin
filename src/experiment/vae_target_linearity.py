import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data import get_mnist, get_cifar10
from src.model import VAE
from src.metric import surrogate, target_linearity
from src.utils import write_json


DATA_DICT = {
    'mnist': (1, 28, get_mnist),
    'cifar10': (3, 32, get_cifar10),
}

# ── Config ────────────────────────────────────────────────────────────────────
DATASET      = sys.argv[1] # 'cifar10'   # 'mnist' | 'cifar10'
DATA_CONFIG = DATA_DICT[DATASET]
IN_CHANNELS  = DATA_CONFIG[0]         # 1 for MNIST, 3 for CIFAR
INPUT_SIZE   = DATA_CONFIG[1]        # 28 for MNIST, 32 for CIFAR
HIDDEN_DIM  = 512
LATENT_DIM  = 32
NUM_HIDDEN  = 2

BATCH_SIZE  = 256
N_SAMPLING  = 10000   # subsample train set to keep gram matrix tractable
PCA_DIM     = 256
BETA        = float(sys.argv[2])

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = Path(__file__).parents[2] / 'params'  / 'vae' / DATASET / str(BETA)
SAVE_DIR = Path(__file__).parents[2] / 'result'  / 'vae' / DATASET / str(BETA)
# ─────────────────────────────────────────────────────────────────────────────


def build_model():
    return VAE(
        in_channels=IN_CHANNELS,
        input_size=INPUT_SIZE,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_hidden=NUM_HIDDEN,
    ).to(DEVICE)


def collect_latents(model, loader):
    """Return (mu, x_flat) for the whole loader."""
    model.eval()
    mus, xs = [], []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            mu, _ = model.encode(inputs)
            mus.append(mu)
            xs.append(inputs.view(inputs.size(0), -1))
    return torch.cat(mus), torch.cat(xs)  # [N, latent_dim], [N, input_dim]


def compute_vae_metrics(model, loader):
    """Compute surrogate and target-linearity of the latent space w.r.t. input."""
    mu, x_flat = collect_latents(model, loader)

    # Target: centered original image
    Y = x_flat.float().to(DEVICE)
    Y = Y - Y.mean(dim=0)   # center across samples, shape [N, input_dim]

    # _, _, V = torch.pca_lowrank(Y, q=10)
    # Y = torch.matmul(Y, V)

    # Feature: latent mean, shape [N, latent_dim]
    H = mu

    # Target linearity: R² of linear regression from H → Y
    tl = target_linearity(Y, H)

    # Surrogate: uses first decoder layer weight as the "next-layer" projection
    # model.decoder[0] is Linear(latent_dim, hidden_dim), weight shape [hidden_dim, latent_dim]
    W = model.decoder[0].weight.data.clone()
    surr, norm = surrogate(Y, H, W, PCA_DIM)

    return surr, tl, norm


def make_loader(dataset, n_sampling=None):
    if n_sampling is not None and n_sampling < len(dataset):
        idx = np.random.choice(len(dataset), size=n_sampling, replace=False)
        dataset = Subset(dataset, idx)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    trainset, _ = DATA_CONFIG[2]()
    train_loader = make_loader(trainset, n_sampling=N_SAMPLING)

    ckpt_files = sorted(CKPT_DIR.glob('epoch*.pt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No epoch checkpoints found in {CKPT_DIR}")

    ret_dict = {
        'checkpoint':             [],
        'surrogate':              [],
        'target_linearity':       [],
        'norm':                   [],
    }

    for ckpt_path in ckpt_files:
        print(f"Loading {ckpt_path.name} ...")
        model = build_model()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        surr, tl, norm = compute_vae_metrics(model, train_loader)

        print(f"  surrogate={surr:.4f}  target_linearity={tl:.4f}  norm={norm:.2f}")

        ret_dict['checkpoint'].append(ckpt_path.name)
        ret_dict['surrogate'].append(round(surr, 6))
        ret_dict['target_linearity'].append(round(tl, 6))
        ret_dict['norm'].append(round(norm, 4))

    save_path = SAVE_DIR / 'tl.json'
    write_json(ret_dict, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
