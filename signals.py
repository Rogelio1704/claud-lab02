"""Indicadores técnicos y regla de confirmación 2 de 3.

Tres indicadores, de tres familias distintas (tendencia, momento,
volatilidad), ninguno basado en volumen. Cada uno traduce el precio en una
dirección discreta {-1, 0, +1}:

1. **Cruce de medias móviles exponenciales (tendencia)**: dirección = signo
   de (EMA_rápida − EMA_lenta), con una banda muerta proporcional al ATR
   para no operar sobre ruido cuando las medias están casi pegadas.
2. **RSI (momento)**: dirección = +1 si el RSI está por encima de
   50 + banda, −1 si está por debajo de 50 − banda, 0 en la zona neutral.
3. **Ruptura de Bandas de Bollinger (volatilidad)**: dirección = +1 si el
   cierre rompe la banda superior, −1 si rompe la inferior, 0 si está dentro
   del canal.

Regla de confirmación (fórmula en el reporte): se abre una posición larga si
al menos 2 de los 3 indicadores valen +1, corta si al menos 2 valen −1, y no
se opera en cualquier otro caso.

Causalidad: todas las funciones de esta sección usan únicamente datos hasta
el cierre de la barra t (medias móviles, EWMA y desviaciones estándar hacia
atrás). La señal confirmada en t se ejecuta en la apertura de t+1; eso lo
implementa `shift_signal_for_execution`, no las funciones de indicador en sí.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class StrategyParams:
    """Hiperparámetros de la estrategia para un régimen y una ventana dados.

    Los rangos de búsqueda (documentados también en el reporte) son:

    - ``ema_fast`` en [5, 40] barras: media rápida de tendencia de corto
      plazo (entre 25 minutos y poco más de 3 horas en datos de 5 min).
    - ``ema_slow`` en [50, 200] barras: media lenta, debe ser mayor que la
      rápida; cubre de ~4 a ~17 horas, suficiente para separar tendencia de
      ruido intradía.
    - ``trend_band_k`` en [0.0, 1.0]: ancho de la banda muerta del cruce de
      medias, en múltiplos de ATR.
    - ``rsi_window`` en [7, 30] barras: ventanas típicas de RSI.
    - ``rsi_band`` en [5, 25] puntos: qué tan lejos de 50 debe estar el RSI
      para considerarse direccional.
    - ``boll_window`` en [10, 60] barras y ``boll_k`` en [1.5, 3.5]
      desviaciones estándar: rangos usuales de Bandas de Bollinger.
    - ``atr_window`` en [7, 30] barras: ventana del ATR usado tanto en la
      banda muerta del cruce de medias como en el stop-loss/take-profit.
    - ``sl_atr_mult`` en [1.0, 4.0] y ``tp_atr_mult`` en [1.0, 6.0]:
      múltiplos de ATR para el stop-loss y el take-profit (TP/SL = precio de
      entrada ± m·ATR, la convención de la clase que funciona para distintos
      regímenes de volatilidad).
    - ``position_fraction`` en [0.1, 1.0]: fracción del capital disponible
      comprometida en cada operación (sizing de fracción fija).
    """

    ema_fast: int
    ema_slow: int
    trend_band_k: float
    rsi_window: int
    rsi_band: float
    boll_window: int
    boll_k: float
    atr_window: int
    sl_atr_mult: float
    tp_atr_mult: float
    position_fraction: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


PARAM_BOUNDS = {
    "ema_fast": (5, 40),
    "ema_slow": (50, 200),
    "trend_band_k": (0.0, 1.0),
    "rsi_window": (7, 30),
    "rsi_band": (5.0, 25.0),
    "boll_window": (10, 60),
    "boll_k": (1.5, 3.5),
    "atr_window": (7, 30),
    "sl_atr_mult": (1.0, 4.0),
    "tp_atr_mult": (1.0, 6.0),
    "position_fraction": (0.1, 1.0),
}


def ema(close: pd.Series, span: int) -> pd.Series:
    """Media móvil exponencial causal (recursiva, ``adjust=False``)."""
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range causal, suavizado con media móvil simple hacia atrás."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, window: int) -> pd.Series:
    """RSI causal con suavizado de Wilder (EWMA con alpha = 1/window)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_value = 100.0 - (100.0 / (1.0 + rs))
    rsi_value = rsi_value.where(avg_loss != 0.0, 100.0)
    return rsi_value


def bollinger_bands(close: pd.Series, window: int, n_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas de Bollinger causales: media y desviación estándar hacia atrás."""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def direction_trend(close: pd.Series, atr_series: pd.Series, fast: int, slow: int, band_k: float) -> pd.Series:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    diff = ema_fast - ema_slow
    band = band_k * atr_series
    sign_diff = np.sign(diff.fillna(0.0)).astype(int)
    direction = pd.Series(0, index=close.index, dtype=int)
    direction = direction.where(diff.abs() <= band, sign_diff)
    direction = direction.where(diff.notna() & band.notna(), 0)
    return direction.astype(int)


def direction_momentum(close: pd.Series, window: int, band: float) -> pd.Series:
    rsi_value = rsi(close, window)
    direction = pd.Series(0, index=close.index, dtype=int)
    direction = direction.where(~(rsi_value > 50.0 + band), 1)
    direction = direction.where(~(rsi_value < 50.0 - band), -1)
    direction = direction.where(rsi_value.notna(), 0)
    return direction.astype(int)


def direction_volatility_breakout(close: pd.Series, window: int, n_std: float) -> pd.Series:
    _, upper, lower = bollinger_bands(close, window, n_std)
    direction = pd.Series(0, index=close.index, dtype=int)
    direction = direction.where(~(close > upper), 1)
    direction = direction.where(~(close < lower), -1)
    direction = direction.where(upper.notna() & lower.notna(), 0)
    return direction.astype(int)


def confirm_signal(dir_trend: pd.Series, dir_momentum: pd.Series, dir_vol: pd.Series,
                    min_votes: int = config.CONFIRMATION_MIN_VOTES) -> pd.Series:
    """Regla de confirmación 2 de 3.

    Fórmula (ver reporte, sección de estrategia):

        votos_largo(t)  = 1{d1(t)=+1} + 1{d2(t)=+1} + 1{d3(t)=+1}
        votos_corto(t)  = 1{d1(t)=-1} + 1{d2(t)=-1} + 1{d3(t)=-1}
        señal(t) = +1  si votos_largo(t)  >= min_votes
        señal(t) = -1  si votos_corto(t) >= min_votes
        señal(t) =  0  en cualquier otro caso
    """
    votes_long = (dir_trend == 1).astype(int) + (dir_momentum == 1).astype(int) + (dir_vol == 1).astype(int)
    votes_short = (dir_trend == -1).astype(int) + (dir_momentum == -1).astype(int) + (dir_vol == -1).astype(int)
    signal = pd.Series(0, index=dir_trend.index, dtype=int)
    signal = signal.where(votes_long < min_votes, 1)
    signal = signal.where(votes_short < min_votes, -1)
    return signal


def compute_indicator_table(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Calcula las tres direcciones, la señal confirmada y el ATR sobre ``df``.

    ``df`` debe ser un tramo contiguo (un episodio o un prefijo de un
    episodio) ordenado cronológicamente, con columnas Open, High, Low,
    Close. El resultado conserva el índice de ``df``.
    """
    close, high, low = df["Close"], df["High"], df["Low"]
    atr_series = atr(high, low, close, params.atr_window)

    dir_trend = direction_trend(close, atr_series, params.ema_fast, params.ema_slow, params.trend_band_k)
    dir_momentum = direction_momentum(close, params.rsi_window, params.rsi_band)
    dir_vol = direction_volatility_breakout(close, params.boll_window, params.boll_k)
    signal = confirm_signal(dir_trend, dir_momentum, dir_vol)

    out = pd.DataFrame(
        {
            "atr": atr_series,
            "dir_trend": dir_trend,
            "dir_momentum": dir_momentum,
            "dir_vol": dir_vol,
            "signal": signal,
        },
        index=df.index,
    )
    return out


def shift_signal_for_execution(signal: pd.Series) -> pd.Series:
    """Desplaza la señal confirmada en t para ejecutarse en la apertura de t+1.

    La fila resultante en la posición t contiene la señal que se confirmó en
    el cierre de t-1 y que, por lo tanto, se ejecuta en la apertura de la
    barra t. La primera barra de la serie siempre queda en 0 (no hay señal
    previa que ejecutar).
    """
    return signal.shift(1).fillna(0).astype(int)
