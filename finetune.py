import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import build_split
from model import EEGMAE


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stratified_subset(y, frac, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ids = np.where(y == c)[0]
        k = max(1, int(round(frac * len(ids))))
        idx += rng.choice(ids, k, replace=False).tolist()
    return np.array(idx)


@torch.no_grad()
def extract(model, X, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model.features(X[i:i + bs].to(device)))
    return torch.cat(out)


@torch.no_grad()
def accuracy_end2end(model, head, X, y, device, bs=128):
    model.eval()
    head.eval()
    correct = 0
    for i in range(0, len(X), bs):
        logits = head(model.features(X[i:i + bs].to(device)))
        correct += (logits.argmax(1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(X)


def run(mode, Xtr, ytr, Xte, yte, cfg, ckpt, steps, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EEGMAE(cfg).to(device)
    if mode != "scratch":
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    head = nn.Linear(cfg.enc_dim, 2).to(device)
    loss_fn = nn.CrossEntropyLoss()

    if mode == "linear":
        ftr = extract(model, Xtr, device)
        fte = extract(model, Xte, device)
        mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-5
        ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
        ytr_d = ytr.to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
        for _ in range(300):
            loss = loss_fn(head(ftr), ytr_d)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (head(fte).argmax(1).cpu() == yte).float().mean().item()

    if mode == "scratch":
        groups = [{"params": model.parameters(), "lr": 5e-4},
                  {"params": head.parameters(), "lr": 5e-4}]
    else:  # finetune
        groups = [{"params": model.parameters(), "lr": 1e-4},
                  {"params": head.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    bs = min(32, len(Xtr))
    for _ in range(steps):
        model.train()
        head.train()
        b = torch.randint(0, len(Xtr), (bs,))
        x, y = Xtr[b].to(device), ytr[b].to(device)
        loss = loss_fn(head(model.features(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return accuracy_end2end(model, head, Xte, yte, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mae.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fracs", default="0.01,0.1,1.0")
    args = ap.parse_args()

    cfg = Config()
    device = get_device()
    fracs = [float(f) for f in args.fracs.split(",")]
    print("device:", device, "| ckpt:", args.ckpt)

    _, Xl, yl = build_split(cfg.train_subjects, cfg, "train", unlabeled=False)
    _, Xt, yt = build_split(cfg.test_subjects, cfg, "test", unlabeled=False)
    Xl_all, yl_all = torch.from_numpy(Xl), torch.from_numpy(yl)
    Xte, yte = torch.from_numpy(Xt), torch.from_numpy(yt)

    modes = ["scratch", "linear", "finetune"]
    res = {m: {str(f): [] for f in fracs} for m in modes}
    for f in fracs:
        for seed in range(args.seeds):
            idx = stratified_subset(yl, f, seed)
            Xs, ys = Xl_all[idx], yl_all[idx]
            for m in modes:
                acc = run(m, Xs, ys, Xte, yte, cfg, args.ckpt,
                          args.steps, seed, device)
                res[m][str(f)].append(acc)
                print(f"frac {f:<5} seed {seed} n={len(idx):<5} {m:<9} acc {acc:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/finetune_results.json", "w") as fh:
        json.dump(res, fh, indent=2)

    print("\nmean +- std test accuracy (held-out subjects)")
    print(f"{'frac':<8}" + "".join(f"{m:<20}" for m in modes))
    for f in fracs:
        row = f"{f:<8}"
        for m in modes:
            a = np.array(res[m][str(f)])
            row += f"{a.mean():.3f} +- {a.std():.3f}".ljust(20)
        print(row)

    plt.figure(figsize=(5.5, 4))
    for m in modes:
        mean = [np.mean(res[m][str(f)]) for f in fracs]
        std = [np.std(res[m][str(f)]) for f in fracs]
        plt.errorbar([f * 100 for f in fracs], mean, yerr=std, marker="o",
                     capsize=3, label=m)
    plt.axhline(0.5, color="gray", ls="--", lw=0.8)
    plt.xscale("log")
    plt.xlabel("% of labeled training data")
    plt.ylabel("test accuracy")
    plt.title("Label efficiency: left vs right imagery")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/label_efficiency.png", dpi=150)
    print("saved results/label_efficiency.png")


if __name__ == "__main__":
    main()
