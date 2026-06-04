from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PatchTSTConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # Attention mask mode:
    # - "reverse": older tokens attend to newer tokens ("far looks near")
    # - "causal": newer tokens attend to older tokens (standard causal)
    # - "none":   no attention mask (bidirectional)
    attn_mode: str = "reverse"
    # Tokenization: non-overlapping blocks along time (interval) axis.
    # Each token is built from `patch_len` consecutive intervals.
    patch_len: int = 5
    # Kept for backward compatibility; ignored in non-overlapping mode.
    patch_stride: int = 5


class PatchTSTRegressor(object):
    def __init__(self, input_size: int, config: Optional[PatchTSTConfig] = None):
        self.input_size = int(input_size)
        self.config = config or PatchTSTConfig()
        self._torch = None
        self.model = None

    def _lazy_init(self):
        if self.model is not None:
            return
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise ImportError(
                "PyTorch is required. Install in Meow env (CPU): "
                "`pip install -U torch --index-url https://download.pytorch.org/whl/cpu`."
            ) from e

        self._torch = torch
        cfg = self.config

        class _Net(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.input_size = input_size
                self.patch_len = cfg.patch_len
                self.patch_stride = cfg.patch_stride

                self.proj = nn.Linear(input_size * cfg.patch_len, cfg.d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.nhead,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
                self.pos_emb = None  # created on first forward

                self.head = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model // 2),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.d_model // 2, 1),
                )

            def _patchify(self, x):
                # x: (B, T, F)
                b, t, f = x.shape
                pl = self.patch_len
                if t < pl:
                    raise ValueError(f"seq_len={t} < patch_len={pl}")
                # Non-overlapping blocks (tokens) that are adjacent in time.
                n_patches = t // pl
                if n_patches <= 0:
                    raise ValueError(f"seq_len={t} produces 0 patches with patch_len={pl}")
                # If T is not divisible by patch_len, drop the oldest remainder so the last
                # token always corresponds to the most recent block.
                start = t - n_patches * pl
                patches = []
                for i in range(n_patches):
                    s = start + i * pl
                    patches.append(x[:, s : s + pl, :].reshape(b, pl * f))
                return torch.stack(patches, dim=1)  # (B, P, pl*F)

            def _build_reverse_causal_mask(self, p: int, device):
                # Allow token i (older) to attend to j >= i (itself and more recent tokens).
                # Disallow attending to the past (j < i).
                # PyTorch bool attn_mask: True means "masked out".
                return torch.tril(torch.ones((p, p), device=device, dtype=torch.bool), diagonal=-1)

            def _build_causal_mask(self, p: int, device):
                # Allow token i (newer) to attend to j <= i (itself and older tokens).
                # Disallow attending to the future (j > i).
                # PyTorch bool attn_mask: True means "masked out".
                return torch.triu(torch.ones((p, p), device=device, dtype=torch.bool), diagonal=1)

            def forward(self, x):
                # x: (B, T, F)
                tok = self._patchify(x)
                tok = self.proj(tok)

                # positional embedding
                if self.pos_emb is None or self.pos_emb.shape[1] != tok.shape[1]:
                    self.pos_emb = torch.nn.Parameter(torch.zeros(1, tok.shape[1], tok.shape[2], device=tok.device))
                    torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

                tok = tok + self.pos_emb

                attn_mode = str(getattr(cfg, "attn_mode", "reverse") or "reverse").lower()
                attn_mask = None
                if attn_mode == "reverse":
                    attn_mask = self._build_reverse_causal_mask(tok.shape[1], tok.device)
                elif attn_mode == "causal":
                    attn_mask = self._build_causal_mask(tok.shape[1], tok.device)
                elif attn_mode in ("none", "bidirectional", "bidir"):
                    attn_mask = None
                else:
                    raise ValueError(f"Unknown attn_mode: {attn_mode} (expected reverse|causal|none)")

                if attn_mask is None:
                    tok = self.encoder(tok)
                else:
                    try:
                        tok = self.encoder(tok, mask=attn_mask)
                    except TypeError:
                        # Older torch versions
                        tok = self.encoder(tok, attn_mask)

                # use last patch token
                out = tok[:, -1, :]
                return self.head(out).squeeze(-1)

        self.model = _Net(self.input_size)

    def parameters(self):
        self._lazy_init()
        return self.model.parameters()

    def to(self, device):
        self._lazy_init()
        self.model.to(device)
        return self

    def train(self):
        self._lazy_init()
        self.model.train()

    def eval(self):
        self._lazy_init()
        self.model.eval()

    def __call__(self, x):
        self._lazy_init()
        return self.model(x)
