from __future__ import annotations

"""
Training schedule utilities.

Separated from flow_matching.py so that config and training scripts can
import them without pulling in JAX/Flax/Optax.
"""


def ema_decay_for(num_steps: int, half_life_frac: float = 0.1) -> float:
    """
    EMA decay such that the half-life is ``half_life_frac`` of training.

    For 50 000 steps with ``half_life_frac=0.1``:
        half_life = 5 000 steps  →  decay = 0.5^(1/5000) ≈ 0.999 861

    Specifying the half-life fraction rather than a raw decay value makes the
    EMA scale correctly when ``num_steps`` changes — there is no need to
    retune the decay constant by hand for different run lengths.

    Parameters
    ----------
    num_steps:
        Total number of training steps (i.e. ``config.train.num_steps``).
    half_life_frac:
        Fraction of ``num_steps`` that is the EMA half-life.
        Default 0.1 (10 % of training).

    Returns
    -------
    float
        EMA decay value in (0, 1).
    """
    half_life_steps = max(1, int(half_life_frac * num_steps))
    return 0.5 ** (1.0 / half_life_steps)
