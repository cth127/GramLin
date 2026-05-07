# based on:
# https://github.com/danielmamay/grokking/blob/main/grokking/data.py

import torch
from torch import nn
import math

torch.set_default_dtype(torch.float64)


DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
    "(x//y)if(y%2==1)else(x-y)": lambda x, y, _: torch.where(y % 2 == 1, x // y, x - y)
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    "x*y": lambda x, y, _: (x, y, x*y),
    **DIVISION_MODULO_OPERATIONS,
    "x^2+y": lambda x, y, _: (x, y, x**2 + y),
    "x^2+y^2": lambda x, y, _: (x, y, x**2 + y**2),
    "x^2+xy+y^2": lambda x, y, _: (x, y, x**2 + x*y + y**2),
    "x^2+xy+y^2+x": lambda x, y, _: (x, y, x**2 + x*y + y**2 + x),
    "x^3+xy": lambda x, y, _: (x, y, x**3 + x*y),
    "x^3+xy^2+x": lambda x, y, _: (x, y, x**3 + x*y**2 + y)
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}


def operation_mod_p_data(operation: str, p: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T

    x, y, z = ALL_OPERATIONS[operation](x, y, p)
    results = z.remainder(p)

    inputs = torch.stack([x, y], dim=1)
    labels = results

    return inputs, labels


def reset_params(model, init_scale=1.0):
    # scaled kaiming uniform code:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.layers[0].weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(model.layers[0].weight, -init_scale*bound, init_scale*bound)

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.layers[1].weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(model.layers[1].weight, -init_scale*bound, init_scale*bound)
    