"""Pruebas de la selección de hiperparámetros: la regla de meseta evita un
pico aislado, y la restricción de operaciones mínimas penaliza configuraciones
poco representativas."""

import numpy as np
import pandas as pd

from src import config, optimize, signals


def _trials_df_sinteticas():
    """6 pruebas sobre un solo parámetro (sl_atr_mult): un pico aislado en 2.0
    (sin ninguna otra prueba cerca) con un Calmar muy alto, y una meseta
    ancha y consistente alrededor de 3.2 con varias pruebas vecinas."""
    filas = []
    valores_pico = {2.0: 200.0}  # pico aislado: ninguna otra prueba a menos de 1.0 de distancia
    valores_meseta = {3.0: 9.0, 3.1: 9.5, 3.2: 9.8, 3.3: 9.4, 3.4: 9.6}  # meseta consistente
    numero = 0
    base = {k: (lo + hi) / 2.0 for k, (lo, hi) in signals.PARAM_BOUNDS.items()}
    for valor_sl, calmar in {**valores_pico, **valores_meseta}.items():
        fila = {f"params_{k}": v for k, v in base.items()}
        fila["params_sl_atr_mult"] = valor_sl
        fila["number"] = numero
        fila["value"] = calmar
        filas.append(fila)
        numero += 1
    return pd.DataFrame(filas)


def test_seleccion_de_meseta_prefiere_la_meseta_sobre_el_pico_aislado():
    trials_df = _trials_df_sinteticas()
    params, info = optimize.select_plateau_params(trials_df, n_bins=10)
    assert info["method"] == "meseta"
    # el valor elegido debe caer en el rango de la meseta (3.0-3.4), no en el pico (2.0)
    assert 2.9 <= params.sl_atr_mult <= 3.5


def test_seleccion_cae_a_argmax_si_no_hay_vecinos():
    filas = []
    base = {k: (lo + hi) / 2.0 for k, (lo, hi) in signals.PARAM_BOUNDS.items()}
    for i, valor_sl in enumerate([1.2, 2.4, 3.6]):
        fila = {f"params_{k}": v for k, v in base.items()}
        fila["params_sl_atr_mult"] = valor_sl
        fila["number"] = i
        fila["value"] = float(i)
        filas.append(fila)
    trials_df = pd.DataFrame(filas)
    params, info = optimize.select_plateau_params(trials_df, n_bins=10)
    assert info["method"] == "argmax_pico_aislado"
    assert np.isclose(params.sl_atr_mult, 3.6)


def test_select_plateau_sin_pruebas_validas_devuelve_none():
    filas = [{f"params_{k}": v for k, v in signals.PARAM_BOUNDS.items()} for _ in range(3)]
    df = pd.DataFrame(filas)
    df["number"] = range(3)
    df["value"] = config.PENALTY_VALUE
    params, info = optimize.select_plateau_params(df)
    assert params is None
    assert info["method"] == "sin_configuracion_valida"


def test_contiguous_true_runs():
    mask = np.array([False, True, True, False, True, False, False, True])
    runs = optimize.contiguous_true_runs(mask)
    assert runs == [(1, 3), (4, 5), (7, 8)]
