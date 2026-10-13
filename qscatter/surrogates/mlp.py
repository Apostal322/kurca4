"""Compact PyTorch MLP surrogate E -> (Re t, Im t) or E -> T.

Low-data training protocol ("early stopping with refit"):
  1. Split the N training points into fit / validation subsets by taking every
     ``val_every``-th interior point as validation (endpoints are always fitted,
     so the model never has to extrapolate). Interleaving keeps both subsets
     spread over the whole energy interval, unlike a random split which can
     leave large gaps with N = 10.
  2. Train on the fit subset with full-batch Adam and record the epoch with the
     best validation loss (patience-based early stopping; best weights kept).
  3. Retrain from the same initialisation on all N points for that number of
     epochs, so no exact evaluation is wasted.
The dense reference grid is never used for training, early stopping or
hyper-parameter choice.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class MLPConfig:
    hidden: tuple[int, ...] = (64, 64)
    # defaults chosen by *validation* loss among 4 candidate settings
    # (scripts/mlp_config_selection.py); none used the reference grid
    activation: str = "silu"  # "tanh" | "silu"
    lr: float = 1e-2
    weight_decay: float = 0.0
    max_epochs: int = 20000
    patience: int = 3000
    min_epochs: int = 500
    val_every: int = 5
    lr_decay_final: float = 1e-3  # cosine decay of lr to lr * this factor
    device: str = "cpu"
    seed: int = 0
    refit: bool = True
    # False: no validation split, train on all N points for max_epochs (ablation)
    early_stopping: bool = True


def set_determinism(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, n_out: int, hidden=(64, 64), activation="tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        layers, d = [], 1
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainLog:
    best_epoch: int = 0
    best_val: float = float("nan")
    final_train_loss: float = float("nan")
    history: list = field(default_factory=list)  # (epoch, train, val) every 50 epochs


class MLPSurrogate:
    """fit(E, Y) / predict(E) wrapper with input normalisation and output scaling.

    Input: E mapped linearly from the known domain to [-1, 1].
    Output: standardised with the training mean / std of each column — justified
    because Re t, Im t and T have O(1) but different ranges; the transform is
    inverted at prediction time.
    """

    name = "mlp"

    def __init__(self, domain, config: MLPConfig | None = None):
        self.lo, self.hi = domain
        self.cfg = config or MLPConfig()
        self.log = TrainLog()

    def _x(self, E):
        x = 2 * (np.asarray(E, dtype=np.float64) - self.lo) / (self.hi - self.lo) - 1
        return torch.tensor(x[:, None], dtype=torch.float64, device=self.cfg.device)

    def _train(self, model, x, y, n_epochs, x_val=None, y_val=None):
        cfg = self.cfg
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=cfg.lr * cfg.lr_decay_final)
        loss_fn = nn.MSELoss()
        best = (float("inf"), 0, None)
        hist = []
        for ep in range(1, n_epochs + 1):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            if x_val is not None:
                with torch.no_grad():
                    vl = loss_fn(model(x_val), y_val).item()
                # with 1-2 validation points an early lucky minimum is likely, so the
                # checkpoint is only eligible after min_epochs
                if ep >= cfg.min_epochs and vl < best[0]:
                    # weights are only needed when we do not refit
                    best = (vl, ep, None if cfg.refit else copy.deepcopy(model.state_dict()))
                if ep % 50 == 0:
                    hist.append((ep, loss.item(), vl))
                if ep >= cfg.min_epochs and ep - best[1] > cfg.patience:
                    break
            elif ep % 50 == 0:
                hist.append((ep, loss.item(), None))
        return best, loss.item(), hist

    def fit(self, E, Y):
        cfg = self.cfg
        set_determinism(cfg.seed)
        E = np.asarray(E, dtype=float)
        Y = np.atleast_2d(np.asarray(Y, dtype=float).T).T
        self.y_mean = Y.mean(0)
        self.y_std = Y.std(0) + 1e-12
        Ys = (Y - self.y_mean) / self.y_std
        n_out = Y.shape[1]
        init = MLP(n_out, cfg.hidden, cfg.activation).double().to(cfg.device)
        init_state = copy.deepcopy(init.state_dict())

        if not cfg.early_stopping:
            model = init
            _, final, hist = self._train(model, self._x(E), torch.tensor(Ys, device=cfg.device), cfg.max_epochs)
            self.log = TrainLog(best_epoch=cfg.max_epochs, final_train_loss=final, history=hist)
            self.model = model.eval()
            return self

        n = E.size
        interior = np.arange(1, n - 1)
        val_idx = interior[cfg.val_every - 1 :: cfg.val_every] if n >= 2 * cfg.val_every else interior[n // 2 : n // 2 + 1]
        fit_idx = np.setdiff1d(np.arange(n), val_idx)

        model = init
        best, _, hist = self._train(
            model, self._x(E[fit_idx]), torch.tensor(Ys[fit_idx], device=cfg.device),
            cfg.max_epochs, self._x(E[val_idx]), torch.tensor(Ys[val_idx], device=cfg.device),
        )
        self.log = TrainLog(best_epoch=best[1], best_val=best[0], history=hist)
        if cfg.refit:
            model = MLP(n_out, cfg.hidden, cfg.activation).double().to(cfg.device)
            model.load_state_dict(init_state)
            set_determinism(cfg.seed + 1)
            _, final, _ = self._train(model, self._x(E), torch.tensor(Ys, device=cfg.device), max(best[1], 1))
            self.log.final_train_loss = final
        else:
            model.load_state_dict(best[2])  # best checkpoint
        self.model = model.eval()
        return self

    def predict(self, E):
        with torch.no_grad():
            ys = self.model(self._x(E)).cpu().numpy()
        return ys * self.y_std + self.y_mean

    # -- persistence --------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(state=self.model.state_dict(), y_mean=self.y_mean, y_std=self.y_std,
                 domain=(self.lo, self.hi), config=asdict(self.cfg), n_out=len(self.y_mean)),
            path,
        )

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        cfg = MLPConfig(**{**d["config"], "hidden": tuple(d["config"]["hidden"])})
        obj = cls(d["domain"], cfg)
        obj.model = MLP(d["n_out"], cfg.hidden, cfg.activation).double()
        obj.model.load_state_dict(d["state"])
        obj.model.eval()
        obj.y_mean, obj.y_std = d["y_mean"], d["y_std"]
        return obj
