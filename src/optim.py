import torch
from torch.optim import Optimizer
from typing import Callable
from contextlib import contextmanager

from src.utils import newtonschulz


class GramFace(Optimizer):
    """
    A wrapper optimizer that wraps any base PyTorch optimizer.

    Workflow per step():
      1. Save W1 (parameters before base optimizer update)
      2. Run base optimizer step → parameters become W2
      3. Compute W2' = transform_fn(W1, W2) for each parameter
      4. Set parameters to W2'

    If requires_2d=True (e.g. for Muon), non-2D params (CNN weights) are
    temporarily reshaped to 2D at __init__ and at each step(), then restored.

    Args:
        base_optimizer_cls: A PyTorch Optimizer class (not instance).
        params: Model parameters.
        transform_fn (Callable): A function (W1, W2) -> W2' applied per parameter.
                                 Defaults to identity (no-op beyond base optimizer).
        requires_2d (bool): If True, reshape non-2D params to 2D around __init__
                            and step(). Use for Muon and similar optimizers.
        **optimizer_kwargs: Passed directly to base_optimizer_cls.
    """

    def __init__(
        self,
        base_optimizer_cls,
        params,
        transform_fn: Callable = None,
        requires_2d: bool = False,
        **optimizer_kwargs,
    ):
        params = list(params)
        self.requires_2d = requires_2d
        self.transform_fn = transform_fn  # None = identity (just base optimizer)

        if requires_2d:
            with _params_viewed_as_2d(params):
                self.base_optimizer = base_optimizer_cls(params, **optimizer_kwargs)
        else:
            self.base_optimizer = base_optimizer_cls(params, **optimizer_kwargs)

        # Share param_groups and state with base optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        # --- Reshape non-2D params if needed ---
        reshaped = {}
        if self.requires_2d:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.requires_grad and p.dim() > 2:
                        reshaped[p] = p.shape
                        p.data = p.data.view(p.shape[0], -1)
                        if p.grad is not None:
                            p.grad = p.grad.view(p.shape[0], -1)

        # --- Save W1 (after reshape, so snapshot is always 2D) ---
        w1_snapshots = {}
        if self.transform_fn is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.requires_grad and len(p.shape) == 2:
                        w1_snapshots[p] = p.data.clone()

        # --- Base optimizer step → W2 ---
        loss = self.base_optimizer.step(closure)

        # --- Compute W2' = transform_fn(W1, W2) and write back ---
        if self.transform_fn is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.requires_grad and p in w1_snapshots:
                        w2_prime = self.transform_fn(w1_snapshots[p], p.data)
                        p.data.copy_(w2_prime)

        # --- Restore original shapes ---
        for p, shape in reshaped.items():
            p.data = p.data.view(shape)
            if p.grad is not None:
                p.grad = p.grad.view(shape)

        return loss

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


@contextmanager
def _params_viewed_as_2d(params):
    """Temporarily reshape non-2D params to 2D in-place for Muon initialization."""
    reshaped = {}
    for p in params:
        if p.dim() > 2:
            reshaped[p] = p.shape
            p.data = p.data.view(p.shape[0], -1)
    try:
        yield
    finally:
        for p, shape in reshaped.items():
            p.data = p.data.view(shape)


def whitening(w1, w2):
    product = w1 @ w2.T
    ns_ = newtonschulz(product, steps=10)
    ret = ns_ @ w2
    return ret