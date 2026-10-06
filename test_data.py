"""Pruebas de limpieza causal y detección de episodios en src/data.py."""

import numpy as np
import pandas as pd

from src import config, data


def _raw_con_hueco_y_nan():
    """20 barras de 5 min, con un hueco de 2 horas entre las barras 9 y 10,
    y una racha de 3 barras con OHLC vacío cerca del inicio."""
    dt1 = pd.date_range("2024-01-01 00:00", periods=10, freq="5min")
    dt2 = pd.date_range("2024-01-01 04:00", periods=10, freq="5min")  # hueco de 2h tras dt1
    dt = dt1.append(dt2)

    n = len(dt)
    close = 100.0 + np.arange(n) * 0.1
    df = pd.DataFrame({
        "Timestamp": dt.astype("int64") // 10**9,
        "Gmtoffset": 0,
        "Datetime": dt,
        "Open": close, "High": close + 0.1, "Low": close - 0.1, "Close": close,
        "Volume": np.nan,
    })
    df.loc[2:4, ["Open", "High", "Low", "Close"]] = np.nan  # racha de 3 barras vacías
    return df


def test_assign_episodes_detecta_el_hueco():
    df_raw = _raw_con_hueco_y_nan()
    episodios = data._assign_episodes(df_raw["Datetime"])
    assert episodios.iloc[:10].nunique() == 1
    assert episodios.iloc[10:].nunique() == 1
    assert episodios.iloc[0] != episodios.iloc[10]


def test_clean_dataset_elimina_filas_vacias_sin_rellenar():
    df_raw = _raw_con_hueco_y_nan()
    limpio = data.clean_dataset(df_raw)
    assert limpio["Close"].isna().sum() == 0
    assert len(limpio) == len(df_raw) - 3  # las 3 filas vacías desaparecen, no se rellenan
    # los valores que sí existían no se modificaron
    assert np.isclose(limpio.iloc[0]["Close"], 100.0)


def test_warmup_marca_no_operable_al_inicio_de_cada_episodio(monkeypatch):
    monkeypatch.setattr(config, "WARMUP_BARS", 3)
    df_raw = _raw_con_hueco_y_nan()
    limpio = data.clean_dataset(df_raw)
    primero_por_episodio = limpio.groupby("episode_id").head(3)
    assert not primero_por_episodio["is_tradable"].any()
    resto = limpio.drop(primero_por_episodio.index)
    assert resto["is_tradable"].all()


def test_auditoria_reporta_el_hueco_y_la_racha_vacia():
    df_raw = _raw_con_hueco_y_nan()
    reporte = data.audit_dataset(df_raw, "sintetico")
    assert reporte.n_ohlc_vacio == 3
    assert reporte.racha_vacio_maxima_barras == 3
    assert len(reporte.huecos_grandes) == 1
