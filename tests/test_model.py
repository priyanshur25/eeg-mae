import torch
from config import Config
from model import EEGMAE

cfg = Config()


def test_shapes():
    m = EEGMAE(cfg)
    x = torch.randn(4, cfg.n_channels, cfg.win_len)
    loss, pred, mask = m(x)
    assert pred.shape == (4, m.L, cfg.patch_len)
    assert mask.shape == (4, m.L)
    assert loss.ndim == 0


def test_mask_ratio_exact():
    m = EEGMAE(cfg)
    x = torch.randn(4, cfg.n_channels, cfg.win_len)
    _, _, mask = m(x, mask_ratio=0.75)
    assert torch.all(mask.sum(1) == int(m.L * 0.75))


def test_kept_tokens_match_mask():
    # tokens hold their own index, so we can check which ones were kept
    B, L, D = 3, 512, 8
    idx = torch.arange(L).float().view(1, L, 1).expand(B, L, D)
    kept, mask, _ = EEGMAE.random_masking(idx, 0.75)
    for b in range(B):
        kept_ids = set(kept[b, :, 0].long().tolist())
        unmasked_ids = set(torch.where(mask[b] == 0)[0].tolist())
        assert kept_ids == unmasked_ids


def test_loss_only_on_masked():
    m = EEGMAE(cfg).eval()
    x = torch.randn(2, cfg.n_channels, cfg.win_len)
    torch.manual_seed(0)
    loss, pred, mask = m(x)
    target = m.patchify(x)
    manual = (((pred - target) ** 2).mean(-1) * mask).sum() / mask.sum()
    assert torch.allclose(loss, manual)


def test_gradients_flow():
    m = EEGMAE(cfg)
    x = torch.randn(2, cfg.n_channels, cfg.win_len)
    m(x)[0].backward()
    assert all(p.grad is not None for p in m.parameters())
