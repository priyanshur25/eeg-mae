import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import build_split, EEGDataset
from model import EEGMAE


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    torch.manual_seed(0)  # same masks every eval -> comparable numbers
    tot, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        loss, _, _ = model(x)
        tot += loss.item() * len(x)
        n += len(x)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--ckpt", default="checkpoints/mae.pt")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    use_amp = device == "cuda"
    print("device:", device, "| amp:", use_amp)

    Xu, _, _ = build_split(cfg.train_subjects, cfg, "train", unlabeled=True)
    _, Xt, yt = build_split(cfg.test_subjects, cfg, "test", unlabeled=False)
    train_loader = DataLoader(EEGDataset(Xu), batch_size=args.bs, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(EEGDataset(Xt, yt), batch_size=128)

    model = EEGMAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup = args.warmup_epochs * steps_per_epoch

    start_epoch = 0
    if args.resume and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log_path = "results/pretrain_log.csv"
    new_log = not (args.resume and os.path.exists(log_path))
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run, n = 0.0, 0
        for x in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = x.to(device)
            lr = lr_at(step, total_steps, warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss, _, _ = model(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            n += 1
            step += 1

        tr = run / n
        va = evaluate(model, val_loader, device)
        sec = time.time() - t0
        print(f"epoch {epoch:3d} | train {tr:.4f} | val {va:.4f} | lr {lr:.2e} | {sec:.0f}s")
        log.writerow([epoch, f"{tr:.5f}", f"{va:.5f}", f"{lr:.3e}", f"{sec:.1f}"])
        log_f.flush()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "cfg": cfg.__dict__}, args.ckpt)

    log_f.close()
    print("done. checkpoint:", args.ckpt)


if __name__ == "__main__":
    main()
