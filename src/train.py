import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, epoch, device='cuda', categorize=True):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    tqdm_loader = tqdm(dataloader, desc=f"[TRAIN (SGD) {epoch} EPOCH]")
    for inputs, targets in tqdm_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
        avg_loss = total_loss / total
        if categorize:
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            accuracy = correct / total
            tqdm_loader.set_postfix(loss=round(avg_loss, 4), accuracy=round(accuracy, 4))
        else:
            accuracy = None
            tqdm_loader.set_postfix(loss=round(avg_loss, 4))
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, epoch, device='cuda', categorize=True):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tqdm_loader = tqdm(dataloader, desc=f"[EVAL {epoch} EPOCH]")
    with torch.no_grad():
        for inputs, targets in tqdm_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            total_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            avg_loss = total_loss / total
            if categorize:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                accuracy = correct / total
                tqdm_loader.set_postfix(loss=round(avg_loss, 4), accuracy=round(accuracy, 4))
            else:
                accuracy = None
                tqdm_loader.set_postfix(loss=round(avg_loss, 4))
    return avg_loss, accuracy


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon, x.view(x.size(0), -1), reduction='sum') / x.size(0)
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(model, dataloader, optimizer, epoch, device='cuda', beta=1.0):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total = 0
    tqdm_loader = tqdm(dataloader, desc=f"[TRAIN (VAE) {epoch} EPOCH]")
    for inputs, _ in tqdm_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(inputs)
        loss, recon_loss, kl_loss = vae_loss(recon, inputs, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        n = inputs.size(0)
        total_loss += loss.item() * n
        total_recon += recon_loss.item() * n
        total_kl += kl_loss.item() * n
        total += n
        tqdm_loader.set_postfix(
            loss=round(total_loss / total, 4),
            recon=round(total_recon / total, 4),
            kl=round(total_kl / total, 4),
        )
    return total_loss / total, total_recon / total, total_kl / total


def evaluate_vae(model, dataloader, epoch, device='cuda', beta=1.0):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total = 0
    tqdm_loader = tqdm(dataloader, desc=f"[EVAL (VAE) {epoch} EPOCH]")
    with torch.no_grad():
        for inputs, _ in tqdm_loader:
            inputs = inputs.to(device)
            recon, mu, logvar = model(inputs)
            loss, recon_loss, kl_loss = vae_loss(recon, inputs, mu, logvar, beta)

            n = inputs.size(0)
            total_loss += loss.item() * n
            total_recon += recon_loss.item() * n
            total_kl += kl_loss.item() * n
            total += n
            tqdm_loader.set_postfix(
                loss=round(total_loss / total, 4),
                recon=round(total_recon / total, 4),
                kl=round(total_kl / total, 4),
            )
    return total_loss / total, total_recon / total, total_kl / total
