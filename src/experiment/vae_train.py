import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data import get_mnist, get_cifar10  # swap to get_cifar10 for CIFAR
from src.model import VAE
from src.train import train_vae, evaluate_vae
from src.utils import write_json


DATA_DICT = {
    'mnist': (1, 28, get_mnist, 1e-4, 10),
    'cifar10': (3, 32, get_cifar10, 1e-4, 30),
}

# ── Config ────────────────────────────────────────────────────────────────────
DATASET      = sys.argv[1] # 'cifar10'   # 'mnist' | 'cifar10'
DATA_CONFIG = DATA_DICT[DATASET]
IN_CHANNELS  = DATA_CONFIG[0]         # 1 for MNIST, 3 for CIFAR
INPUT_SIZE   = DATA_CONFIG[1]        # 28 for MNIST, 32 for CIFAR

HIDDEN_DIM   = 512
LATENT_DIM   = 32
NUM_HIDDEN   = 2

BATCH_SIZE   = 128
LR           = DATA_CONFIG[3]
EPOCHS       = DATA_CONFIG[4]
BETA         = float(sys.argv[2]) # 1.0       # weight on KL term (beta-VAE: try 2-4)

CKPT_EVERY   = 1 # max(1, EPOCHS // 10)  # save ~10 checkpoints total

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
LOSS_DIR     = Path(__file__).parents[2] / 'result' / 'vae' / DATASET / str(BETA)
CKPT_DIR     = Path(__file__).parents[2] / 'params' / 'vae' / DATASET / str(BETA)
SAVE_DIR     = LOSS_DIR / 'vis'
# ─────────────────────────────────────────────────────────────────────────────


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    trainset, testset = DATA_CONFIG[2]()   # <- swap dataset here
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = VAE(
        in_channels=IN_CHANNELS,
        input_size=INPUT_SIZE,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_hidden=NUM_HIDDEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_history = []
    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_recon, train_kl = train_vae(model, train_loader, optimizer, epoch, DEVICE, BETA)
        eval_loss,  eval_recon,  eval_kl  = evaluate_vae(model, test_loader, epoch, DEVICE, BETA)

        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}) | "
            f"eval loss {eval_loss:.4f} (recon {eval_recon:.4f}, kl {eval_kl:.4f})"
        )

        loss_history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_recon': train_recon, 'train_kl': train_kl,
            'eval_loss':  eval_loss,  'eval_recon':  eval_recon,  'eval_kl':  eval_kl,
        })

        # Save reconstructions every 5 epochs
        if epoch % 1 == 0:
            _save_reconstructions(model, test_loader, epoch)

        # Save checkpoint every CKPT_EVERY epochs and at the final epoch
        if epoch % CKPT_EVERY == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), CKPT_DIR / f'epoch{epoch:03d}.pt')

        # Checkpoint best model
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_DIR / 'best.pt')

    os.rename(CKPT_DIR / 'best.pt', CKPT_DIR / f'best_{best_epoch}.pt')
    torch.save(model.state_dict(), CKPT_DIR / 'final.pt')
    write_json(loss_history, LOSS_DIR / 'loss.json')
    print(f"Done. Best eval loss: {best_loss:.4f}")


def _save_reconstructions(model, loader, epoch, n=16):
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(loader))
        inputs = inputs[:n].to(DEVICE)
        recon, _, _ = model(inputs)
        recon = recon.view_as(inputs)
        comparison = torch.cat([inputs.cpu(), recon.cpu()])
        save_image(comparison, SAVE_DIR / f'recon_epoch{epoch:03d}.png', nrow=n)


if __name__ == '__main__':
    main()
