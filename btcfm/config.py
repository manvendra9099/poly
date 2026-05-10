from __future__ import annotations

"""
Configuration dataclasses loaded from YAML.

All config objects are frozen after construction so they can be hashed and
logged reliably. Config hash is logged at the start of each training run for
reproducibility.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    context_length: int = 240   # L
    horizon: int = 60           # H
    norm_window: int = 1440     # rolling z-score trailing window (minutes)
    max_gap_bars: int = 5       # skip windows containing gaps > this many synthetic bars


@dataclass(frozen=True)
class ModelConfig:
    encoder_dim: int = 128
    encoder_heads: int = 4
    encoder_layers: int = 4
    encoder_dropout: float = 0.1
    velocity_hidden: int = 256
    velocity_layers: int = 6
    time_emb_dim: int = 128


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    batch_size: int = 256
    num_steps: int = 50000
    # EMA half-life as a fraction of total training steps.  The actual decay
    # is computed at runtime via btcfm.model.training.ema_decay_for() so that
    # the EMA automatically scales to different run lengths.
    ema_half_life_frac: float = 0.1
    # Precision mode: "auto" selects bf16 on A100/H100, fp32 on V100/CPU.
    # "bf16" / "fp32" override auto-detection. "fp16" is accepted but
    # experimental — do not use for the first production run.
    precision: str = "auto"
    log_every: int = 500
    val_every: int = 500
    checkpoint_every: int = 5000
    seed: int = 42


@dataclass(frozen=True)
class SamplerConfig:
    solver: Literal["heun", "euler"] = "heun"
    n_steps: int = 50
    n_samples: int = 1000


@dataclass(frozen=True)
class VerifyConfig:
    # # DESIGN: stride of 5 minutes for test-set verification on CPU.
    # At n_samples=1000 and 50 Heun steps, one ensemble takes several seconds.
    # 15 days × 1440 / 5 = 4320 forecasts rather than 21600. GPU users can set to 1.
    forecast_stride: int = 5
    lead_times_for_diag: tuple[int, ...] = (1, 5, 15, 30, 60)


@dataclass(frozen=True)
class BTCFMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)

    def config_hash(self) -> str:
        """SHA-256 of the JSON-serialised config for run identification."""
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(path: str | Path) -> BTCFMConfig:
    """
    Load a YAML config file and return a frozen BTCFMConfig.

    Unknown keys are silently ignored; missing keys fall back to dataclass defaults.
    """
    raw = yaml.safe_load(Path(path).read_text())

    def _pick(raw_dict: dict, cls) -> dict:
        return {k: v for k, v in raw_dict.items() if k in cls.__dataclass_fields__}

    data_cfg = DataConfig(**_pick(raw.get("data", {}), DataConfig))
    model_cfg = ModelConfig(**_pick(raw.get("model", {}), ModelConfig))
    train_cfg = TrainConfig(**_pick(raw.get("train", {}), TrainConfig))
    sampler_cfg = SamplerConfig(**_pick(raw.get("sampler", {}), SamplerConfig))
    verify_cfg_raw = raw.get("verify", {})
    # Convert list -> tuple for lead_times_for_diag if present
    if "lead_times_for_diag" in verify_cfg_raw:
        verify_cfg_raw["lead_times_for_diag"] = tuple(verify_cfg_raw["lead_times_for_diag"])
    verify_cfg = VerifyConfig(**_pick(verify_cfg_raw, VerifyConfig))

    cfg = BTCFMConfig(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        sampler=sampler_cfg,
        verify=verify_cfg,
    )
    logger.info("Loaded config from %s (hash=%s)", path, cfg.config_hash())
    return cfg
