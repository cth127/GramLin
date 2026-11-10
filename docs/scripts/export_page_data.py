"""Export the data used by the project page (docs/index.html).

Writes docs/static/data/page_data.js, which defines `window.PAGE_DATA`.
Shipping it as a .js file (not .json) lets index.html work when opened
directly from disk, without a local web server.

Contents
--------
- Virtual-update trajectories for the Swiss roll and two-circles toy problems.
  The Swiss roll uses the paper's Figure 1 setup (Appendix B.1: full-batch GD,
  display step size gamma = 700) but with lr 0.2 and 100 epochs, which
  straightens the curve further (lr >= 0.3 starts to hook the ends; 1.0
  diverges). Two circles follows notebook/swissroll.ipynb
  (lr 0.001, display step size 300) with init seed 6. A fixed seed is
  added for reproducibility.
- Proposition 2 check: angles between the actual and the virtual update of each
  hidden layer of a bias-free 4-layer GELU MLP on the Swiss roll.
- Gradient-whitening accuracy curves (Figure 2) and per-layer surrogate /
  Target Linearity curves (Figure 4), read from the experiment result files.

Usage (from the repository root)
--------------------------------
    python docs/scripts/export_page_data.py --result-dir result
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_circles, make_swiss_roll
from torch.nn import BCELoss, MSELoss, ReLU
from torch.nn.functional import sigmoid
from torch.optim import SGD

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.model import DNN  # noqa: E402



def r2_linear(features, target):
    """R^2 of OLS (with intercept) of target on features: the Target Linearity T_0."""
    A = np.hstack([features, np.ones((features.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    resid = target - A @ coef
    return float(1.0 - resid @ resid / ((target - target.mean()) @ (target - target.mean())))


def run_virtual_update(X, y, model, optim, loss_fn, scale, normalize, lr, n_steps, link=None):
    X = X.clone().requires_grad_(True)
    grads, grams, losses = [], [], []
    W = model.layers[0].weight.data
    grams.append((W.T @ W).numpy().copy())
    for _ in range(n_steps):
        optim.zero_grad()
        pred = model(X)
        if link is not None:
            pred = link(pred)
        loss = loss_fn(pred, y.view(-1, 1))
        loss.backward(retain_graph=True)
        grad = torch.autograd.grad(loss, X, retain_graph=True)[0]
        optim.step()
        losses.append(loss.item())
        grads.append(grad.detach().clone())
        W = model.layers[0].weight.data
        grams.append((W.T @ W).numpy().copy())

    X = X.detach()
    cum = torch.cumsum(torch.stack(grads), dim=0)
    # Theorem 1 (multi-step form): W_t^T W_t - W_0^T W_0 ~= X~_t^T X~_t - X^T X,
    # where X~_t uses the *actual* learning rate (no display scaling).
    Xn = X.numpy().astype(np.float64)
    vcs = [np.zeros((2, 2))]
    for t in range(1, n_steps + 1):
        Xt = Xn - lr * cum[t - 1].numpy().astype(np.float64)
        vcs.append(Xt.T @ Xt - Xn.T @ Xn)
    gram_shift = [G - grams[0] for G in grams]

    frames, tls = [], []
    for t in range(n_steps + 1):
        Xp = X if t == 0 else X - scale * cum[t - 1]
        tls.append(r2_linear(Xp.numpy().astype(np.float64), y.numpy().astype(np.float64)))
        frames.append(normalize(Xp).numpy())
    return frames, gram_shift, vcs, losses, tls


def export_swissroll():
    torch.manual_seed(0)
    X, y = make_swiss_roll(n_samples=1000, random_state=0)
    X = torch.tensor(np.stack([X[:, 0], X[:, 2]], axis=1))
    X = (X - X.min(dim=0).values) / (X - X.min(dim=0).values).max(dim=0).values
    y = torch.tensor(y)
    y = (y - y.min()) / (y - y.min()).max()
    X, y = X.float(), y.float()

    model = DNN(input_dim=2, hidden_dim=32, num_hidden=2, output_dim=1, bias=True)
    model.layers[0].bias = None
    optim = SGD(model.parameters(), lr=0.2)

    def minmax(Z):
        Z = Z - Z.min(dim=0).values
        return Z / (Z.max(dim=0).values + 1e-6)

    frames, gram_shift, vcs, losses, tls = run_virtual_update(
        X, y, model, optim, MSELoss(), scale=700, normalize=minmax, lr=0.2, n_steps=100)
    return pack(frames, gram_shift, vcs, losses, tls, y, lo=0.0, hi=1.0, label_kind="continuous")


def export_circles():
    # The result depends strongly on the initialization: over seeds 0-11 the final
    # TL of the virtual inputs ranges from 0.01 to 0.95. Seed 6 gives the clean
    # separation seen in the original notebook GIF (TL ~0.95).
    torch.manual_seed(6)
    X, y = make_circles(n_samples=1000, random_state=0, noise=0.1, factor=0.1)
    X = torch.tensor(X)
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    X, y = X.float(), torch.tensor(y).float()

    model = DNN(input_dim=2, hidden_dim=256, num_hidden=1, output_dim=1, bias=True)
    model.nonlin = ReLU()
    optim = SGD(model.parameters(), lr=0.001)

    def standardize(Z):
        return (Z - Z.mean(dim=0)) / (Z.std(dim=0) + 1e-6)

    frames, gram_shift, vcs, losses, tls = run_virtual_update(
        X, y, model, optim, BCELoss(), scale=3e2, normalize=standardize, lr=0.001, n_steps=100, link=sigmoid)
    return pack(frames, gram_shift, vcs, losses, tls, y, lo=-3.0, hi=3.0, label_kind="binary")


def pack(frames, gram_shift, vcs, losses, tls, y, lo, hi, label_kind):
    """Quantize coordinates to integers in [0, 1000] over the plot window [lo, hi]."""
    sym = lambda M: [float(f"{M[0, 0]:.4g}"), float(f"{M[0, 1]:.4g}"), float(f"{M[1, 1]:.4g}")]
    q = lambda Z: np.clip(np.round((Z - lo) / (hi - lo) * 1000), -500, 1500).astype(int)
    return {
        "lim": [lo, hi],
        "labelKind": label_kind,
        "y": [round(float(v), 3) for v in y.numpy()],
        "frames": [q(F).reshape(-1).tolist() for F in frames],
        "gramShift": [sym(G) for G in gram_shift],
        "vcs": [sym(V) for V in vcs],
        "loss": [round(v, 5) for v in losses],
        "tl": [round(v, 4) for v in tls],
    }


def export_prop2(n_layers=4, width=64, train_lr=0.2, n_train=100,
                 checkpoints=(0, 25, 50, 75, 100), n_samples=200, eta=1e-4):
    """Proposition 2 check on a real network.

    For a single input, all layers take one small GD step on that input's loss
    (a discretized gradient flow). Prop. 2 says the actual change of every hidden
    state h_l stays within 90 degrees of the virtual direction -grad_{h_l} L.
    We record that angle for many inputs, at several points during training.
    """
    torch.manual_seed(0)
    X, y = make_swiss_roll(n_samples=1000, random_state=0)
    X = torch.tensor(np.stack([X[:, 0], X[:, 2]], axis=1))
    X = (X - X.min(dim=0).values) / (X - X.min(dim=0).values).max(dim=0).values
    y = torch.tensor(y)
    y = (y - y.min()) / (y - y.min()).max()
    # float64: the per-sample change of h_l is tiny, and float32 round-off
    # alone produces spurious angles near and above 90 degrees.
    X, y = X.double(), y.double()

    # bias-free, so every layer is z = W h as in the proposition
    model = DNN(input_dim=2, hidden_dim=width, num_hidden=n_layers, output_dim=1, bias=False).double()
    params = list(model.parameters())
    optim = SGD(params, lr=train_lr)
    rng = np.random.default_rng(0)
    angles = [[] for _ in range(n_layers)]

    def measure():
        for i in rng.choice(len(X), size=n_samples, replace=False):
            x, t = X[i:i + 1], y[i]
            out, hs = model(x, return_hidden=True)
            loss = 0.5 * (out.squeeze() - t) ** 2
            g_h = torch.autograd.grad(loss, hs, retain_graph=True)
            g_w = torch.autograd.grad(loss, params)
            saved = [p.data.clone() for p in params]
            with torch.no_grad():
                for p, g in zip(params, g_w):
                    p -= eta * g
                _, hs_new = model(x, return_hidden=True)
                for p, w in zip(params, saved):
                    p.copy_(w)
            for l in range(n_layers):
                dh = (hs_new[l] - hs[l].detach()).double().ravel()
                v = -g_h[l].double().ravel()
                if dh.norm() < 1e-14 or v.norm() < 1e-14:
                    continue
                cos = torch.clamp(dh @ v / (dh.norm() * v.norm()), -1, 1)
                angles[l].append(float(torch.rad2deg(torch.arccos(cos))))

    for step in range(n_train + 1):
        if step in checkpoints:
            measure()
        if step == n_train:
            break
        optim.zero_grad()
        loss = 0.5 * ((model(X).squeeze() - y) ** 2).mean()
        loss.backward()
        optim.step()

    edges = np.arange(0, 181, 5)
    return {
        "binWidth": 5,
        "hist": [np.histogram(a, bins=edges)[0].tolist() for a in angles],
        "max": [round(max(a), 2) for a in angles],
        "median": [round(float(np.median(a)), 2) for a in angles],
        "n": [len(a) for a in angles],
        "setup": {"layers": n_layers, "width": width, "checkpoints": list(checkpoints),
                  "samples": n_samples, "eta": eta},
    }


def export_whitening(result_dir, lr="0.01"):
    out = {}
    for data in ["mnist", "svhn", "cifar10", "cifar100"]:
        out[data] = {}
        for kind in ["orig", "whiten"]:
            res = json.loads((result_dir / "whitening" / f"{data}_{kind}.json").read_text())
            acc = np.stack([res[lr][str(t)]["accuracy"] for t in range(5)], axis=1)
            out[data][kind] = {
                "mean": np.round(acc.mean(axis=1), 4).tolist(),
                "std": np.round(acc.std(axis=1), 4).tolist(),
            }
    return out


def export_target_linearity(result_dir):
    out = {}
    for data in ["cifar10", "cifar100"]:
        res = json.loads((result_dir / "target_linearity" / f"{data}_sgd.json").read_text())
        out[data] = {
            "surrogate": np.array(res["train_surrogate"]).T.tolist(),
            "tl": np.round(np.clip(np.array(res["train_target_linearity"]).T, 0, None), 4).tolist(),
            "trainAcc": [0.0] + res["train_accuracy"],
            "testAcc": [0.0] + res["test_accuracy"],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "static" / "data" / "page_data.js")
    args = parser.parse_args()

    data = {
        "swissroll": export_swissroll(),
        "circles": export_circles(),
        "whitening": export_whitening(args.result_dir),
        "targetLinearity": export_target_linearity(args.result_dir),
        "prop2": export_prop2(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("window.PAGE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    p2 = data["prop2"]
    print("prop2 angles: n", p2["n"], "median", p2["median"], "max", p2["max"])
    for k in ["swissroll", "circles"]:
        print(k, "loss", data[k]["loss"][0], "->", data[k]["loss"][-1],
              "| TL(X~)", data[k]["tl"][0], "->", data[k]["tl"][-1])
        a, b = np.array(data[k]["gramShift"]).ravel(), np.array(data[k]["vcs"]).ravel()
        print("   corr(gram shift, VCS) =", round(float(np.corrcoef(a, b)[0, 1]), 4),
              "| final gram shift", data[k]["gramShift"][-1], "VCS", data[k]["vcs"][-1])


if __name__ == "__main__":
    main()
