from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters."""

    n_features: int = 94

    context_len: int = 256

    attn_window: int = 256

    n_layers: int = 4

    n_heads: int = 8

    model_dim: int = 256

    ffn_dim: int = 704

    dropout: float = 0.2

    stock_emb_dim: int = 32

    input_scale: float = 1.0

    cross_stock_attn: bool = True

    target_horizons: tuple = ("fret1", "fret6", "fret12", "fret24")

@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    lr: float = 2e-4

    weight_decay: float = 0.5

    grad_clip: float = 1.0

    batch_size: int = 256

    n_epochs: int = 2

    warmup_ratio: float = 0.1

    lr_scheduler_eta_min: float = 5e-5

@dataclass
class PreprocessingConfig:
    """Feature preprocessing parameters and fitted state.

    Fields marked [FITTED] are computed from data by fit_preprocessing().
    Do NOT set them manually — they will be overwritten.
    """

    preprocessing_fit_days: int = 20

    eps: float = 1e-8

    feat_p01: float | None = None

    feat_p99: float | None = None

    feat_log_mask: list[bool] | None = None

    feat_mean: float | None = None

    feat_std: float | None = None

    y_std: float | None = None

    vol20_idx: int | None = None

    vol20_floor: float = 1e-8

MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
