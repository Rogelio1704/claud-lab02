"""Pruebas de la selección de hiperparámetros: la regla de meseta evita un
pico aislado, y la restricción de operaciones mínimas penaliza configuraciones
poco representativas."""

import numpy as np
import pandas as pd

from src import config, metrics, optimize, signals


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


def test_objetivo_corregido_prefiere_perder_menos():
    """Defecto 1.1: el Calmar sin condicionar premia la ruina (perder casi
    todo da un Calmar MENOS negativo que perder poco, porque el retorno
    anualizado tiene piso en -100% mientras el drawdown sigue creciendo).
    La función objetivo corregida debe preferir, entre dos configuraciones
    perdedoras, la que pierde menos."""
    dt = pd.date_range("2024-01-01", periods=3, freq="5min")
    pierde_poco = pd.Series([1_000_000.0, 995_000.0, 990_000.0], index=dt)
    pierde_casi_todo = pd.Series([1_000_000.0, 100_000.0, 10_000.0], index=dt)

    # demuestra el defecto: el Calmar crudo ordena al revés de lo económico
    calmar_poco = metrics.calmar_ratio(pierde_poco)
    calmar_mucho = metrics.calmar_ratio(pierde_casi_todo)
    assert calmar_mucho > calmar_poco  # el Calmar crudo preferiría perder casi todo

    # la función objetivo corregida ordena correctamente
    obj_poco = optimize.compute_objective(pierde_poco)
    obj_mucho = optimize.compute_objective(pierde_casi_todo)
    assert obj_poco > obj_mucho
    assert -1.0 <= obj_mucho <= 0.0
    assert -1.0 <= obj_poco <= 0.0


def test_objetivo_usa_calmar_cuando_el_retorno_es_positivo():
    dt = pd.date_range("2024-01-01", periods=4, freq="7D")  # semanas, no minutos: evita un exponente de anualización extremo
    # sube, cae (hay drawdown) y termina arriba del inicio: retorno positivo
    equity = pd.Series([1_000_000.0, 1_100_000.0, 900_000.0, 1_050_000.0], index=dt)
    objetivo = optimize.compute_objective(equity)
    calmar_directo = metrics.calmar_ratio(equity)
    assert np.isclose(objetivo, calmar_directo)
    assert objetivo > 0
