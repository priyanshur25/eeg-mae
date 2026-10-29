from config import Config
from data import build_split, EEGDataset

cfg = Config()

Xu, Xl, yl = build_split(cfg.train_subjects, cfg, "train", unlabeled=True)
_, Xt, yt = build_split(cfg.test_subjects, cfg, "test", unlabeled=False)

print("pretrain windows :", Xu.shape)
print("train labeled    :", Xl.shape, "class counts:", [(yl == c).sum() for c in (0, 1)])
print("test labeled     :", Xt.shape, "class counts:", [(yt == c).sum() for c in (0, 1)])
print("mean/std of one window:", Xl[0].mean(), Xl[0].std())

ds = EEGDataset(Xl, yl)
x, y = ds[0]
print("sample:", x.shape, x.dtype, y)
