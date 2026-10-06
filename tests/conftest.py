"""Configuración de pytest: hace importable el paquete ``src`` sin instalar
el proyecto, y fija la semilla aleatoria para que las pruebas sean
reproducibles."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


@pytest.fixture(autouse=True)
def _fijar_semilla():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    yield


def make_synthetic_episode(n=400, start="2024-01-01", seed=config.RANDOM_SEED, episode_id=1):
    """Crea un episodio sintético (OHLC + columnas auxiliares) para pruebas,
    sin depender de los CSV reales ni del umbral de calentamiento real."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0, scale=0.003, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.001, n)))

    dt = pd.date_range(start=start, periods=n, freq="5min")
    df = pd.DataFrame({
        "Timestamp": dt.astype("int64") // 10**9,
        "Gmtoffset": 0,
        "Datetime": dt,
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": np.nan,
    })
    df["episode_id"] = episode_id
    df["bar_in_episode"] = np.arange(n)
    df["is_tradable"] = True
    return df
