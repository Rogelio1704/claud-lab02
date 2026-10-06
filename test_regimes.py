"""Pruebas de causalidad del detector de régimen: la etiqueta en t no cambia
al agregar datos posteriores a t, ni en las variables ni en la predicción."""

import numpy as np

from src import regimes
from tests.conftest import make_synthetic_episode

VENTANA_PRUEBA = 30  # ventana más chica que la de producción, solo para que las pruebas corran rápido


def test_causalidad_features_de_regimen_en_tres_puntos():
    df = make_synthetic_episode(n=200)
    completas = regimes.compute_regime_features(df, window=VENTANA_PRUEBA, ema_fast_span=5, ema_slow_span=15)

    for t in [60, 120, 190]:
        parciales = regimes.compute_regime_features(df.iloc[: t + 1], window=VENTANA_PRUEBA,
                                                      ema_fast_span=5, ema_slow_span=15)
        fila_completa = completas.iloc[t].fillna(-999)
        fila_parcial = parciales.iloc[t].fillna(-999)
        assert np.allclose(fila_completa.to_numpy(), fila_parcial.to_numpy()), f"Causalidad violada en t={t}"


def test_etiqueta_de_regimen_no_cambia_al_agregar_datos_futuros():
    """Con el modelo YA ajustado (solo con el pasado), la etiqueta de una
    barra pasada debe ser idéntica sin importar cuántas barras futuras se
    agreguen a la serie sobre la que se predice."""
    df_largo = make_synthetic_episode(n=250)
    features_largo = regimes.compute_regime_features(df_largo, window=VENTANA_PRUEBA, ema_fast_span=5, ema_slow_span=15)

    ajuste = features_largo.iloc[:150].dropna()
    modelo = regimes.fit_regime_model(ajuste)

    codigos_hasta_180 = modelo.predict_codes(features_largo.iloc[:180])
    codigos_hasta_250 = modelo.predict_codes(features_largo)

    comunes = codigos_hasta_180.index
    assert (codigos_hasta_180.to_numpy() == codigos_hasta_250.loc[comunes].to_numpy()).all()


def test_etiquetas_mantienen_cadencia_de_actualizacion():
    df = make_synthetic_episode(n=200)
    features = regimes.compute_regime_features(df, window=VENTANA_PRUEBA, ema_fast_span=5, ema_slow_span=15)
    ajuste = features.dropna()
    modelo = regimes.fit_regime_model(ajuste)
    codigos = modelo.predict_codes(features)

    etiquetas = regimes.labels_with_update_cadence(codigos, update_bars=10)
    # dentro de cada bloque de 10 barras, la etiqueta debe ser constante
    for inicio in range(0, len(etiquetas) - 10, 10):
        bloque = etiquetas.iloc[inicio: inicio + 10]
        assert bloque.nunique() == 1
