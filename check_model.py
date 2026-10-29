import torch
from config import Config
from model import EEGMAE

cfg = Config()
torch.manual_seed(cfg.seed)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

model = EEGMAE(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params / 1e6:.2f}M | tokens per window: {model.L}")

x = torch.randn(8, cfg.n_channels, cfg.win_len, device=device)
loss, pred, mask = model(x)
print("loss:", loss.item())
print("pred shape:", tuple(pred.shape), "| mask shape:", tuple(mask.shape))
print("masked fraction:", mask.mean().item())
print("features shape:", tuple(model.features(x).shape))

loss.backward()
grads = [p.grad is not None for p in model.parameters()]
print("all params got gradients:", all(grads))
