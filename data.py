import os
import numpy as np
import mne
import torch
from mne.datasets import eegbci
from torch.utils.data import Dataset

from config import Config

mne.set_log_level("ERROR")


def zscore(x):
    """x: (n, C, T) -> per-window, per-channel z-score."""
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True)
    return (x - m) / (s + 1e-6)


def load_runs(subject, cfg):
    files = eegbci.load_data(subject, list(cfg.runs), verbose=False)
    raws = []
    for f in files:
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        eegbci.standardize(raw)
        raw.filter(cfg.l_freq, cfg.h_freq, verbose=False)
        raws.append(raw)
    return raws


def sliding_windows(data, win, stride):
    """data: (C, T) -> (n, C, win)"""
    starts = range(0, data.shape[1] - win + 1, stride)
    return np.stack([data[:, s:s + win] for s in starts])


def labeled_epochs(raw, win):
    """Cut a `win`-sample epoch at each T1 (left=0) / T2 (right=1) onset."""
    events, _ = mne.events_from_annotations(
        raw, event_id={"T1": 1, "T2": 2}, verbose=False
    )
    data = raw.get_data()
    X, y = [], []
    for onset, _, ev in events:
        seg = data[:, onset:onset + win]
        if seg.shape[1] == win:
            X.append(seg)
            y.append(ev - 1)
    return X, y


def build_split(subjects, cfg, name, unlabeled=True):
    """Returns (X_unlab or None, X_lab, y_lab). Cached to disk as .npz."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = os.path.join(cfg.cache_dir, f"{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        xu = d["X_unlab"] if unlabeled else None
        return xu, d["X_lab"], d["y_lab"]

    X_unlab, X_lab, y_lab = [], [], []
    for s in subjects:
        print(f"subject {s}")
        for raw in load_runs(s, cfg):
            if unlabeled:
                X_unlab.append(sliding_windows(raw.get_data(), cfg.win_len, cfg.stride))
            xs, ys = labeled_epochs(raw, cfg.win_len)
            X_lab += xs
            y_lab += ys

    X_lab = zscore(np.stack(X_lab)).astype(np.float32)
    y_lab = np.array(y_lab, dtype=np.int64)
    if unlabeled:
        X_unlab = zscore(np.concatenate(X_unlab)).astype(np.float32)
    else:
        X_unlab = np.zeros((0,), dtype=np.float32)

    np.savez(path, X_unlab=X_unlab, X_lab=X_lab, y_lab=y_lab)
    return (X_unlab if unlabeled else None), X_lab, y_lab


class EEGDataset(Dataset):
    """Returns (x, y) if y is given, else just x. x: (C, T) float32."""

    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = None if y is None else torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i] if self.y is None else (self.X[i], self.y[i])
