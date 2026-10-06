"""Pruebas de causalidad de indicadores y de la regla de confirmación 2 de 3."""

import numpy as np
import pandas as pd

from src import signals
from tests.conftest import make_synthetic_episode

PARAMS = signals.StrategyParams(
    ema_fast=8, ema_slow=34, trend_band_k=0.2,
    rsi_window=14, rsi_band=10.0,
    boll_window=20, boll_k=2.0,
    atr_window=14, sl_atr_mult=2.0, tp_atr_mult=3.0,
    position_fraction=0.3,
)


def test_causalidad_indicadores_en_tres_puntos():
    """Recalcular la señal sobre df.iloc[:t+1] debe dar el mismo valor en t
    que calcularla sobre la serie completa, para varios valores de t."""
    df = make_synthetic_episode(n=300)
    tabla_completa = signals.compute_indicator_table(df, PARAMS)

    for t in [100, 150, 250]:
        tabla_parcial = signals.compute_indicator_table(df.iloc[: t + 1], PARAMS)
        fila_completa = tabla_completa.iloc[t].fillna(-999)
        fila_parcial = tabla_parcial.iloc[t].fillna(-999)
        assert (fila_completa == fila_parcial).all(), f"Causalidad violada en t={t}"


def test_shift_signal_ejecuta_en_t_mas_1():
    df = make_synthetic_episode(n=50)
    tabla = signals.compute_indicator_table(df, PARAMS)
    ejecutada = signals.shift_signal_for_execution(tabla["signal"])
    assert ejecutada.iloc[0] == 0
    assert (ejecutada.iloc[1:].to_numpy() == tabla["signal"].iloc[:-1].to_numpy()).all()


def test_confirmacion_un_indicador_no_abre_posicion():
    idx = pd.RangeIndex(5)
    dir_trend = pd.Series([1, 0, 0, -1, 0], index=idx)
    dir_momentum = pd.Series([0, 0, 1, 0, 0], index=idx)
    dir_vol = pd.Series([0, 1, 0, 0, -1], index=idx)
    señal = signals.confirm_signal(dir_trend, dir_momentum, dir_vol)
    # en cada barra, a lo más un indicador vota en cada dirección: nunca debe abrirse posición
    assert (señal == 0).all()


def test_confirmacion_dos_indicadores_abre_largo_y_corto():
    idx = pd.RangeIndex(2)
    dir_trend = pd.Series([1, -1], index=idx)
    dir_momentum = pd.Series([1, -1], index=idx)
    dir_vol = pd.Series([0, 0], index=idx)
    señal = signals.confirm_signal(dir_trend, dir_momentum, dir_vol)
    assert señal.iloc[0] == 1
    assert señal.iloc[1] == -1


def test_confirmacion_tres_indicadores_en_conflicto_no_opera():
    idx = pd.RangeIndex(1)
    dir_trend = pd.Series([1], index=idx)
    dir_momentum = pd.Series([-1], index=idx)
    dir_vol = pd.Series([0], index=idx)
    señal = signals.confirm_signal(dir_trend, dir_momentum, dir_vol)
    assert señal.iloc[0] == 0
