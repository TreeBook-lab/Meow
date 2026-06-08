import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from log import log
from parameters import MODEL_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)

def _precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0):
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(theta) / dim))
    angles = position * div_term.unsqueeze(0)
    return angles.cos(), angles.sin()

def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.repeat_interleave(cos, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.repeat_interleave(sin, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    x_half = x.shape[-1] // 2
    x1, x2 = x[..., :x_half], x[..., x_half:]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rot * sin

class SelfAttention(nn.Module):
    def __init__(self, *, model_dim: int, n_heads: int, dropout: float,
                 context_len: int, attn_window: int = 64):
        super().__init__()
        if model_dim % n_heads != 0:
            raise ValueError("model_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.context_len = context_len
        self.attn_window = attn_window

        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=True)
        self.out = nn.Linear(model_dim, model_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        rope_cos, rope_sin = _precompute_rope_freqs(self.head_dim, context_len)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        if t > self.context_len:
            raise ValueError(f"seq len {t} exceeds context_len {self.context_len}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rotary_emb(q, self.rope_cos[:t], self.rope_sin[:t])
        k = _apply_rotary_emb(k, self.rope_cos[:t], self.rope_sin[:t])

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        mask = _causal_mask(t, x.device)
        if self.attn_window > 0:
            row = torch.arange(t, device=x.device).unsqueeze(1)
            col = torch.arange(t, device=x.device).unsqueeze(0)
            mask = mask | ((row - col) > self.attn_window)
        mask = mask.view(1, 1, t, t)
        if padding_mask is not None:
            mask = mask | padding_mask.view(b, 1, 1, t)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        return self.out(out.transpose(1, 2).contiguous().view(b, t, d))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=True)
        self.w_up = nn.Linear(d_model, d_ff, bias=True)
        self.w_out = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_up(x))


class DecoderBlock(nn.Module):
    def __init__(self, *, model_dim: int, n_heads: int, ffn_dim: int, dropout: float,
                 context_len: int, attn_window: int = 64):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.attn = SelfAttention(
            model_dim=model_dim, n_heads=n_heads, dropout=dropout,
            context_len=context_len, attn_window=attn_window)
        self.ffn = SwiGLUFFN(model_dim, ffn_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.norm1(x), padding_mask=padding_mask))
        x = x + self.resid_dropout(self.ffn(self.norm2(x)))
        return x

class CrossStockAttention(nn.Module):

    def __init__(self, model_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            model_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.transpose(0, 1)
        key_pad = None
        if padding_mask is not None:
            key_pad = padding_mask.transpose(0, 1)
        x, _ = self.attn(x, x, x, key_padding_mask=key_pad)
        x = x.transpose(0, 1)
        return residual + self.dropout(x)


class SparseAttentionRegressor(nn.Module):
    def __init__(self, cfg, max_stocks: int = 500):
        super().__init__()
        self.cfg = cfg

        self.feat_gate = nn.Parameter(torch.ones(cfg.n_features))

        self.feat_proj = nn.Linear(cfg.n_features, cfg.model_dim, bias=True)
        self.stock_emb = nn.Linear(max_stocks, cfg.stock_emb_dim, bias=False)
        self.stock_proj = nn.Linear(cfg.stock_emb_dim, cfg.model_dim, bias=False)
        self.feat_norm = RMSNorm(cfg.model_dim)
        self.drop = nn.Dropout(cfg.dropout)

        self.cross_stock = None
        if cfg.cross_stock_attn:
            self.cross_stock = CrossStockAttention(
                cfg.model_dim, cfg.n_heads, cfg.dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                model_dim=cfg.model_dim, n_heads=cfg.n_heads,
                ffn_dim=cfg.ffn_dim, dropout=cfg.dropout,
                context_len=cfg.context_len, attn_window=cfg.attn_window)
            for _ in range(cfg.n_layers)
        ])

        self.norm_f = RMSNorm(cfg.model_dim)
        self.heads = nn.ModuleDict({
            h: nn.Linear(cfg.model_dim, 1, bias=True)
            for h in cfg.target_horizons})
        self.lin_skips = nn.ModuleDict({
            h: nn.Linear(cfg.n_features, 1, bias=True)
            for h in cfg.target_horizons})
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x_raw: torch.Tensor, stock_ids: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                horizon: str | None = None) -> dict | torch.Tensor:
        b, t, _ = x_raw.shape

        gated = x_raw * self.feat_gate.sigmoid()

        x = self.feat_proj(gated)
        stock_onehot = F.one_hot(
            stock_ids, num_classes=self.stock_emb.in_features).float()
        s = self.stock_proj(self.stock_emb(stock_onehot))
        x = x + s.unsqueeze(1)
        x = self.feat_norm(x)
        x = self.drop(x)

        if self.cross_stock is not None:
            x = self.cross_stock(x, padding_mask)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        x = self.norm_f(x)
        scale = self.log_scale.exp()

        if horizon is not None:
            return self.heads[horizon](x).squeeze(-1) * scale + \
                self.lin_skips[horizon](gated).squeeze(-1)
        return {h: self.heads[h](x).squeeze(-1) * scale +
                self.lin_skips[h](gated).squeeze(-1)
                for h in self.heads}

class MeowModel:
    def __init__(self):
        self.cfg = MODEL_CONFIG
        self.tcfg = TRAINING_CONFIG
        self.pcfg = PREPROCESSING_CONFIG

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SparseAttentionRegressor(self.cfg).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.tcfg.lr,
            weight_decay=self.tcfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        self.scheduler = None
        self._sym_to_id: dict[str, int] = {}

    def _preprocess_x(self, batch_x: np.ndarray) -> np.ndarray:
        batch_x = np.clip(batch_x, self.pcfg.feat_p01, self.pcfg.feat_p99)
        if self.pcfg.feat_log_mask.any():
            batch_x[:, :, self.pcfg.feat_log_mask] = np.log1p(
                batch_x[:, :, self.pcfg.feat_log_mask])
        batch_x = (batch_x - self.pcfg.feat_mean) / self.pcfg.feat_std
        batch_x = batch_x * self.cfg.input_scale
        return batch_x

    @staticmethod
    def _symbols(xdf):
        return xdf.index.get_level_values("symbol").unique()

    @staticmethod
    def _sequence(xdf_or_ydf, sym):
        return xdf_or_ydf.loc[sym].sort_index().to_numpy().astype(np.float32)

    @staticmethod
    def _clean(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _symbol_ids(self, syms) -> np.ndarray:
        ids = []
        for s in syms:
            if s not in self._sym_to_id:
                self._sym_to_id[s] = len(self._sym_to_id)
            ids.append(self._sym_to_id[s])
        return np.array(ids, dtype=np.int64)

    def _prepare_batch(self, xdf, ydf, syms):
        n_horizons = len(self.cfg.target_horizons)
        seqs_x = [self._clean(self._sequence(xdf, s)) for s in syms]
        seqs_y = [self._clean(self._sequence(ydf, s)) for s in syms]
        max_ctx = self.cfg.context_len
        seqs_x = [s[-max_ctx:] if len(s) > max_ctx else s for s in seqs_x]
        seqs_y = [s[-max_ctx:] if len(s) > max_ctx else s for s in seqs_y]
        lengths = [len(s) for s in seqs_x]
        max_len = max(lengths)

        b, f = len(syms), seqs_x[0].shape[-1]
        batch_x = np.zeros((b, max_len, f), dtype=np.float32)
        batch_y = np.zeros((b, max_len, n_horizons), dtype=np.float32)
        mask = np.ones((b, max_len), dtype=bool)

        for i, (sx, sy, L) in enumerate(zip(seqs_x, seqs_y, lengths)):
            batch_x[i, :L] = sx
            batch_y[i, :L] = sy
            mask[i, :L] = False

        del seqs_x, seqs_y

        if self.pcfg.feat_mean is not None:
            batch_x = self._preprocess_x(batch_x)

        sym_ids = self._symbol_ids(syms)
        x_t = torch.from_numpy(batch_x).to(self.device)
        return (
            x_t,
            torch.from_numpy(batch_y).to(self.device),
            torch.from_numpy(mask).to(self.device),
            torch.from_numpy(sym_ids).to(self.device),
        )

    def fit_preprocessing(self, xdf, ydf):
        primary_y = ydf["fret12"].to_numpy().ravel()
        all_y = self._clean(primary_y)
        all_x = self._clean(xdf.to_numpy())

        self.pcfg.vol20_idx = list(xdf.columns).index("vol20")
        vol20_vals = np.abs(all_x[:, self.pcfg.vol20_idx])
        self.pcfg.vol20_floor = max(float(np.percentile(vol20_vals, 5)), 1e-8)
        vol20_safe = np.clip(vol20_vals, self.pcfg.vol20_floor, None) + 1e-8
        self.pcfg.y_std = 1.0

        vol_normed_std = float(np.std(all_y / vol20_safe))
        log.inf("vol20 index: {} | p05 floor: {:.6f} | "
                "std(return/vol20_clipped) = {:.4f}".format(
                    self.pcfg.vol20_idx, self.pcfg.vol20_floor, vol_normed_std))

        self.pcfg.feat_p01 = np.percentile(all_x, 1, axis=0).astype(np.float32)
        self.pcfg.feat_p99 = np.percentile(all_x, 99, axis=0).astype(np.float32)

        clipped = np.clip(all_x, self.pcfg.feat_p01, self.pcfg.feat_p99)
        feat_min = clipped.min(axis=0)
        mu = clipped.mean(axis=0)
        sigma = clipped.std(axis=0) + 1e-8
        skew = ((clipped - mu) ** 3).mean(axis=0) / (sigma ** 3)
        self.pcfg.feat_log_mask = (feat_min >= 0) & (skew > 1.5)

        if self.pcfg.feat_log_mask.any():
            clipped[:, self.pcfg.feat_log_mask] = np.log1p(
                clipped[:, self.pcfg.feat_log_mask])

        self.pcfg.feat_mean = clipped.mean(axis=0, keepdims=True).astype(np.float32)
        raw_std = clipped.std(axis=0)
        std_floor = max(float(np.median(raw_std)) * 0.01, 1e-4)
        self.pcfg.feat_std = np.maximum(raw_std, std_floor).astype(np.float32)

        feat_names = list(xdf.columns)
        log.inf("Preprocessing fitted on {} rows | "
                "clip range: [{:.4f}, {:.4f}] → [{:.4f}, {:.4f}] | "
                "log1p features: {} | input_scale: {}".format(
                    len(all_y),
                    self.pcfg.feat_p01.min(), self.pcfg.feat_p01.max(),
                    self.pcfg.feat_p99.min(), self.pcfg.feat_p99.max(),
                    [feat_names[i] for i, m in enumerate(self.pcfg.feat_log_mask) if m],
                    self.cfg.input_scale))

        actual_nf = xdf.shape[1]
        if actual_nf != self.cfg.n_features:
            raise RuntimeError(
                "Feature count mismatch: data has {} features but model "
                "expects {}. Update ModelConfig.n_features or "
                "MeowFeatureGenerator.feature_names().".format(
                    actual_nf, self.cfg.n_features))

        del all_x, clipped, vol20_vals, vol20_safe

    def partial_fit(self, xdf, ydf):
        if self.pcfg.y_std is None:
            raise RuntimeError(
                "Preprocessing not fitted. Call fit_preprocessing() first.")

        self.model.train()
        syms = self._symbols(xdf)
        horizons = list(self.cfg.target_horizons)

        if self.pcfg.vol20_idx is not None:
            vol20_arr = np.abs(self._clean(xdf["vol20"].to_numpy().ravel()))
            vol20_arr = np.clip(vol20_arr, self.pcfg.vol20_floor, None) + 1e-8
            ydf = ydf.copy()
            for h in horizons:
                ydf[h] = ydf[h].to_numpy().ravel() / vol20_arr

        for start in range(0, len(syms), self.tcfg.batch_size):
            batch_syms = syms[start:start + self.tcfg.batch_size]
            x, y, mask, sym_ids = self._prepare_batch(xdf, ydf, batch_syms)
            y = y / self.pcfg.y_std

            with torch.amp.autocast("cuda"):
                preds = self.model(
                    x, stock_ids=sym_ids, padding_mask=mask)
                valid = ~mask

                total_loss = 0.0
                for hi, h in enumerate(horizons):
                    p_v = preds[h][valid]
                    y_v = y[:, :, hi][valid]
                    y_var = y_v.var() + 1e-8
                    loss_mse = F.mse_loss(p_v, y_v) / y_var
                    p_c = p_v - p_v.mean()
                    y_c = y_v - y_v.mean()
                    cov = (p_c * y_c).mean()
                    loss_corr = -cov / ((p_v.std() + 1e-8) * (y_v.std() + 1e-8))
                    total_loss = total_loss + 3.0 * loss_mse + 0.2 * loss_corr

                gate_penalty = 5e-4 * self.model.feat_gate.sigmoid().sum()
                loss = total_loss + gate_penalty

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip)
                self.optimizer.step()

            del x, y, mask, sym_ids, preds, valid, loss

        if self.scheduler is not None:
            self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def set_scheduler(self, steps_per_epoch: int, n_epochs: int):
        t_max = steps_per_epoch * n_epochs
        warmup_steps = int(t_max * self.tcfg.warmup_ratio)
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max - warmup_steps,
            eta_min=self.tcfg.lr_scheduler_eta_min)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_steps])
        log.inf("LR scheduler: warmup {} steps + CosineAnnealing, "
                "T_max={}".format(warmup_steps, t_max))

    def predict(self, xdf, denormalize=True, horizon="fret12"):
        self.model.eval()
        syms = self._symbols(xdf)
        all_preds = []

        with torch.no_grad():
            for start in range(0, len(syms), self.tcfg.batch_size):
                batch_syms = syms[start:start + self.tcfg.batch_size]
                seqs_x = [self._clean(self._sequence(xdf, s))
                          for s in batch_syms]
                lengths = [len(s) for s in seqs_x]
                max_len = max(lengths)

                b, f = len(batch_syms), seqs_x[0].shape[-1]
                batch_x = np.zeros((b, max_len, f), dtype=np.float32)
                mask = np.ones((b, max_len), dtype=bool)
                for i, (sx, L) in enumerate(zip(seqs_x, lengths)):
                    batch_x[i, :L] = sx
                    mask[i, :L] = False
                del seqs_x

                vol20_raw = None
                if self.pcfg.vol20_idx is not None:
                    vol20_vals = np.abs(batch_x[:, :, self.pcfg.vol20_idx])
                    vol20_raw = np.clip(
                        vol20_vals, self.pcfg.vol20_floor, None) + 1e-8

                if self.pcfg.feat_mean is not None:
                    batch_x = self._preprocess_x(batch_x)

                x = torch.from_numpy(batch_x).to(self.device)
                mask_t = torch.from_numpy(mask).to(self.device)
                sym_ids = torch.from_numpy(
                    self._symbol_ids(batch_syms)).to(self.device)

                with torch.amp.autocast("cuda"):
                    p = self.model(
                        x, stock_ids=sym_ids, padding_mask=mask_t,
                        horizon=horizon)
                p = p.float().cpu().numpy()
                del x, mask_t, sym_ids

                if denormalize:
                    if self.pcfg.y_std is not None:
                        p = p * self.pcfg.y_std
                    if vol20_raw is not None:
                        p = p * vol20_raw

                for i, L in enumerate(lengths):
                    all_preds.append(p[i, :L])

        return np.concatenate(all_preds)