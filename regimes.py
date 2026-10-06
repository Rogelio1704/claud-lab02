"""Detección dinámica de régimen de mercado (Nivel B).

Método elegido: **K-means** sobre una matriz de 5 variables causales,
calculadas con una ventana móvil de 1 semana (``config.REGIME_WINDOW_BARS``).
Se prefiere K-means sobre un HMM porque evita por completo el riesgo de usar
la ruta de Viterbi (que reetiqueta el pasado con información futura): con
K-means, una vez fijos el escalador y los centroides, la etiqueta de
cualquier barra depende solo de sus propias variables, calculadas con datos
hasta esa barra. También es más simple de defender frente a un modelo
basado en reglas fijas (K-means encuentra la separación óptima en los datos
de entrenamiento en vez de que el equipo fije percentiles a mano).

Variables de régimen (todas con ventana móvil de 1 semana, 2016 barras):

1. ``vol_realized``: desviación estándar de los retornos de 5 minutos
   (nivel de volatilidad).
2. ``atr_norm``: ATR promedio de la semana, normalizado por el precio
   (segundo proxy de volatilidad, en unidades de precio en vez de retornos).
3. ``trend_strength``: |EMA rápida − EMA lenta| / (vol_realized · precio):
   qué tan lejos está el precio de su media lenta, en unidades de
   volatilidad (fuerza de tendencia).
4. ``r2_trend``: R² de una regresión lineal del precio contra el tiempo
   dentro de la ventana (qué tan "recta" ha sido la tendencia).
5. ``autocorr_lag1``: autocorrelación de orden 1 de los retornos de 5
   minutos (negativa sugiere reversión a la media, positiva sugiere
   momentum/tendencia).

Causalidad: el escalador y los centroides de K-means se ajustan una sola vez
por ventana de entrenamiento del walk-forward, usando solo filas hasta el
final de esa ventana. Para las barras posteriores (ventana de prueba, test)
solo se EVALÚA (transform + predict) con ese modelo ya fijo; nunca se vuelve
a ajustar con datos futuros. La etiqueta se recalcula cada
``config.REGIME_UPDATE_BARS`` barras (2 horas) y se mantiene constante entre
actualizaciones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config

FEATURE_COLUMNS = ["vol_realized", "atr_norm", "trend_strength", "r2_trend", "autocorr_lag1"]

# Ventanas fijas (no optimizadas) de las medias usadas solo para medir fuerza
# de tendencia dentro de la matriz de features de régimen. Son internas al
# detector de régimen y distintas de las ventanas de los indicadores de la
# estrategia (esas sí se optimizan por régimen en optimize.py).
_REGIME_EMA_FAST = 50
_REGIME_EMA_SLOW = 200


def compute_regime_features(df_episode: pd.DataFrame, window: int = config.REGIME_WINDOW_BARS,
                             ema_fast_span: int = _REGIME_EMA_FAST, ema_slow_span: int = _REGIME_EMA_SLOW) -> pd.DataFrame:
    """Matriz de features de régimen, causal, sobre un episodio contiguo.

    ``ema_fast_span``/``ema_slow_span`` solo se exponen para poder probar la
    función con series sintéticas cortas; en producción siempre se usan las
    constantes del módulo.
    """
    close = df_episode["Close"]
    high = df_episode["High"]
    low = df_episode["Low"]
    returns = close.pct_change()

    vol_realized = returns.rolling(window, min_periods=window).std(ddof=0)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr_norm = true_range.rolling(window, min_periods=window).mean() / close

    ema_fast = close.ewm(span=ema_fast_span, adjust=False, min_periods=ema_fast_span).mean()
    ema_slow = close.ewm(span=ema_slow_span, adjust=False, min_periods=ema_slow_span).mean()
    vol_price = (vol_realized * close).replace(0.0, np.nan)
    trend_strength = (ema_fast - ema_slow).abs() / vol_price

    time_idx = pd.Series(np.arange(len(df_episode)), index=df_episode.index, dtype=float)
    corr_trend = close.rolling(window, min_periods=window).corr(time_idx)
    r2_trend = corr_trend ** 2

    autocorr_lag1 = returns.rolling(window, min_periods=window).corr(returns.shift(1))

    features = pd.DataFrame(
        {
            "vol_realized": vol_realized,
            "atr_norm": atr_norm,
            "trend_strength": trend_strength,
            "r2_trend": r2_trend,
            "autocorr_lag1": autocorr_lag1,
        },
        index=df_episode.index,
    )
    return features


@dataclass
class RegimeModel:
    """Escalador + K-means ya ajustados, con el mapeo determinista a nombre."""

    scaler: StandardScaler
    kmeans: KMeans
    cluster_to_code: dict  # {cluster_id_kmeans: 0/1/2 según config.REGIME_NAMES}
    silhouette: float
    n_fit_rows: int

    def predict_codes(self, features: pd.DataFrame) -> pd.Series:
        """Predice el código de régimen (0,1,2) para filas con features completas.

        Las filas con algún NaN (calentamiento insuficiente) devuelven -1.
        """
        valid = features.notna().all(axis=1)
        codes = pd.Series(-1, index=features.index, dtype=int)
        if valid.any():
            x = self.scaler.transform(features.loc[valid, FEATURE_COLUMNS].to_numpy())
            clusters = self.kmeans.predict(x)
            codes.loc[valid] = [self.cluster_to_code[c] for c in clusters]
        return codes


def _label_clusters(features_fit: pd.DataFrame, cluster_labels: np.ndarray) -> dict:
    """Regla determinista para nombrar los 3 clusters encontrados.

    1. El cluster con mayor volatilidad realizada promedio es "crisis".
    2. Entre los dos restantes, el de mayor fuerza de tendencia promedio
       (``trend_strength``) es "tendencia"; el otro es "reversion".
    """
    df = features_fit.copy()
    df["cluster"] = cluster_labels
    medias = df.groupby("cluster")[FEATURE_COLUMNS].mean()

    crisis_cluster = medias["vol_realized"].idxmax()
    restantes = [c for c in medias.index if c != crisis_cluster]
    tendencia_cluster = medias.loc[restantes, "trend_strength"].idxmax()
    reversion_cluster = [c for c in restantes if c != tendencia_cluster][0]

    mapping = {
        tendencia_cluster: config.REGIME_NAMES.index("tendencia"),
        reversion_cluster: config.REGIME_NAMES.index("reversion"),
        crisis_cluster: config.REGIME_NAMES.index("crisis"),
    }
    return mapping


def fit_regime_model(features: pd.DataFrame, seed: int = config.RANDOM_SEED) -> RegimeModel:
    """Ajusta escalador + K-means sobre filas con features completas (sin NaN).

    Solo debe recibir ``features`` calculadas con datos hasta el final de la
    ventana de entrenamiento vigente; nunca con datos futuros.
    """
    valid = features.dropna(subset=FEATURE_COLUMNS)
    if len(valid) < config.N_REGIMES * 10:
        raise ValueError(
            f"Muy pocas filas con features completas ({len(valid)}) para ajustar {config.N_REGIMES} regímenes."
        )

    scaler = StandardScaler()
    x = scaler.fit_transform(valid[FEATURE_COLUMNS].to_numpy())

    kmeans = KMeans(n_clusters=config.N_REGIMES, n_init=10, random_state=seed)
    cluster_labels = kmeans.fit_predict(x)

    mapping = _label_clusters(valid, cluster_labels)

    sil = _silhouette_safe(x, cluster_labels)

    return RegimeModel(
        scaler=scaler, kmeans=kmeans, cluster_to_code=mapping,
        silhouette=sil, n_fit_rows=len(valid),
    )


def _silhouette_safe(x: np.ndarray, labels: np.ndarray, max_sample: int = 20000,
                      seed: int = config.RANDOM_SEED) -> float:
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return np.nan
    if len(x) > max_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_sample, replace=False)
        return float(silhouette_score(x[idx], labels[idx]))
    return float(silhouette_score(x, labels))


def labels_with_update_cadence(codes: pd.Series, update_bars: int = config.REGIME_UPDATE_BARS) -> pd.Series:
    """Mantiene la etiqueta constante entre actualizaciones cada ``update_bars``.

    Se "muestrea" el código de régimen cada ``update_bars`` barras (a partir
    de la primera fila válida) y se propaga hacia adelante hasta la próxima
    muestra. Es estrictamente causal: la etiqueta vigente en t solo usa
    información calculada hasta t.
    """
    codes = codes.copy()
    n = len(codes)
    sampled = pd.Series(-1, index=codes.index, dtype=int)
    positions = np.arange(n)
    sample_mask = (positions % update_bars) == 0
    sampled.loc[sample_mask] = codes.loc[sample_mask]
    held = sampled.replace(-1, np.nan).ffill()
    held = held.where(sample_mask.cumsum() > 0, -1)
    return held.fillna(-1).astype(int)


def regime_duration_stats(regime_codes: pd.Series, bar_minutes: int = config.BAR_MINUTES) -> pd.DataFrame:
    """Duración promedio (horas) y número de episodios por régimen, más transiciones."""
    valid = regime_codes[regime_codes >= 0]
    if valid.empty:
        return pd.DataFrame(columns=["regimen", "duracion_media_horas", "n_episodios", "n_transiciones"])

    change = valid != valid.shift(1)
    run_id = change.cumsum()
    run_sizes = valid.groupby(run_id).size()
    run_regime = valid.groupby(run_id).first()
    tabla = pd.DataFrame({"regimen": run_regime, "n_barras": run_sizes})
    tabla["duracion_horas"] = tabla["n_barras"] * bar_minutes / 60.0

    resumen = tabla.groupby("regimen")["duracion_horas"].agg(["mean", "count"]).rename(
        columns={"mean": "duracion_media_horas", "count": "n_episodios"}
    )
    n_transiciones = int(change.sum()) - 1
    resumen["n_transiciones_total"] = n_transiciones
    resumen["regimen_nombre"] = [config.REGIME_NAMES[i] for i in resumen.index]
    return resumen.reset_index()


def time_share_by_regime(regime_codes: pd.Series) -> pd.Series:
    valid = regime_codes[regime_codes >= 0]
    if valid.empty:
        return pd.Series(dtype=float)
    share = valid.value_counts(normalize=True).sort_index()
    share.index = [config.REGIME_NAMES[i] for i in share.index]
    return share
