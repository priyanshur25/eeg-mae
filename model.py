import torch
import torch.nn as nn


def make_blocks(dim, depth, heads):
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=4 * dim,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class EEGMAE(nn.Module):
    """Masked autoencoder over (channel, time-patch) tokens of EEG."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.P = cfg.patch_len
        self.C = cfg.n_channels
        self.n_t = cfg.win_len // self.P
        self.L = self.C * self.n_t
        D, Dd = cfg.enc_dim, cfg.dec_dim

        # encoder
        self.patch_embed = nn.Linear(self.P, D)
        self.chan_emb = nn.Parameter(torch.zeros(1, self.C, 1, D))
        self.time_emb = nn.Parameter(torch.zeros(1, 1, self.n_t, D))
        self.encoder = make_blocks(D, cfg.enc_depth, cfg.enc_heads)
        self.enc_norm = nn.LayerNorm(D)

        # decoder
        self.dec_embed = nn.Linear(D, Dd)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, Dd))
        self.dec_chan_emb = nn.Parameter(torch.zeros(1, self.C, 1, Dd))
        self.dec_time_emb = nn.Parameter(torch.zeros(1, 1, self.n_t, Dd))
        self.decoder = make_blocks(Dd, cfg.dec_depth, cfg.dec_heads)
        self.dec_norm = nn.LayerNorm(Dd)
        self.dec_head = nn.Linear(Dd, self.P)

        for p in (self.chan_emb, self.time_emb, self.mask_token,
                  self.dec_chan_emb, self.dec_time_emb):
            nn.init.trunc_normal_(p, std=0.02)

    # ---------- helpers ----------
    def patchify(self, x):
        """(B, C, T) -> (B, L, P) with L = C * n_t, ordered channel-major."""
        B = x.shape[0]
        return x.reshape(B, self.C, self.n_t, self.P).reshape(B, self.L, self.P)

    def embed(self, x):
        """(B, C, T) -> (B, L, D) tokens with channel + time embeddings."""
        B = x.shape[0]
        tok = self.patch_embed(self.patchify(x))                  # (B, L, D)
        tok = tok.reshape(B, self.C, self.n_t, -1)
        tok = tok + self.chan_emb + self.time_emb
        return tok.reshape(B, self.L, -1)

    @staticmethod
    def random_masking(tokens, mask_ratio):
        """Keep a random subset of tokens per sample.
        Returns kept tokens, mask (1 = masked, 0 = kept), ids_restore."""
        B, L, D = tokens.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        kept = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, L, device=tokens.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return kept, mask, ids_restore

    # ---------- MAE pieces ----------
    def forward_encoder(self, x, mask_ratio):
        tok = self.embed(x)
        kept, mask, ids_restore = self.random_masking(tok, mask_ratio)
        kept = self.enc_norm(self.encoder(kept))
        return kept, mask, ids_restore

    def forward_decoder(self, latent, ids_restore):
        B = latent.shape[0]
        x = self.dec_embed(latent)                                 # (B, n_keep, Dd)
        n_mask = self.L - x.shape[1]
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)                     # (B, L, Dd)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x = x.reshape(B, self.C, self.n_t, -1)
        x = x + self.dec_chan_emb + self.dec_time_emb
        x = x.reshape(B, self.L, -1)
        x = self.dec_norm(self.decoder(x))
        return self.dec_head(x)                                    # (B, L, P)

    def forward(self, x, mask_ratio=None):
        """Returns (loss, pred, mask). Loss is MSE on masked patches only."""
        if mask_ratio is None:
            mask_ratio = self.cfg.mask_ratio
        target = self.patchify(x)
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        per_token = ((pred - target) ** 2).mean(dim=-1)            # (B, L)
        loss = (per_token * mask).sum() / mask.sum()
        return loss, pred, mask

    # ---------- downstream ----------
    def features(self, x):
        """No masking: full-token encoder pass, mean-pooled -> (B, enc_dim)."""
        h = self.enc_norm(self.encoder(self.embed(x)))
        return h.mean(dim=1)
