from tqdm import tqdm
import torch


def compute_vc(model, train_loader, loss_fn, learning_rate, device='cuda', categorize=True):
    ret = None
    model.eval()
    tqdm_loader = tqdm(train_loader, desc="[Computing VC]")

    for inputs, targets in tqdm_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad_(True)

        outputs = model(inputs)
        loss_value = loss_fn(outputs, targets)
        J = torch.autograd.grad(loss_value, inputs, retain_graph=True)[0]

        with torch.no_grad():
            virtual_update = inputs - learning_rate * J
            inputs = inputs.view(inputs.size(0), 1, -1)
            virtual_update = virtual_update.view(inputs.size(0), 1, -1)
            if categorize:
                virtual_update_covariance = torch.einsum('Bac,Bbc->ab', virtual_update, virtual_update)
                input_covariance = torch.einsum('Bac,Bbc->ab', inputs, inputs)
            else:
                virtual_update_covariance = torch.einsum('Bca,Bcb->ab', virtual_update, virtual_update)
                input_covariance = torch.einsum('Bca,Bcb->ab', inputs, inputs)

            if ret is None:
                ret = (virtual_update_covariance - input_covariance)
            else:
                ret += (virtual_update_covariance - input_covariance)
    return ret
