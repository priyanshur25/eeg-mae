from dataclasses import dataclass


@dataclass
class Config:
    # data
    sfreq: float = 160.0
    n_channels: int = 64
    win_len: int = 320            # 2 s
    stride: int = 160             # 1 s hop for unlabeled pretraining windows
    l_freq: float = 0.5
    h_freq: float = 40.0
    runs: tuple = (4, 8, 12)      # left/right fist imagery
    train_subjects: tuple = tuple(range(1, 31))
    test_subjects: tuple = tuple(range(31, 41))
    cache_dir: str = "cache"
    seed: int = 42
