"""Carga, auditoría y limpieza causal de los datos de BTCUSDT.

Reglas de diseño (declaradas también en el reporte):

1. Los CSV crudos en ``data/`` nunca se modifican en disco. Toda limpieza
   ocurre en memoria, dentro de este módulo.
2. Nunca se interpola ni se rellena hacia adelante/atrás: una barra con
   OHLC vacío se elimina. Rellenar inventaría información que no existió.
3. Un "hueco grande" es una diferencia de marca de tiempo CRUDA (antes de
   eliminar filas vacías) mayor a ``config.GAP_THRESHOLD_MINUTES``. Los tres
   huecos grandes del proyecto (~65 min y ~48 h en train, ~122 días en test)
   parten los datos en "episodios": tramos continuos dentro de los cuales sí
   es válido calcular indicadores y régimen de forma causal. Ninguna barra
   usa información de un episodio anterior después de un hueco grande.
4. Las primeras ``config.WARMUP_BARS`` barras de cada episodio quedan
   marcadas como no operables (``is_tradable = False``): no hay garantía de
   que los indicadores ni el modelo de régimen tengan suficiente historia
   confiable antes de ese punto. Esto respeta la causalidad porque un
   episodio que empieza justo donde terminó el train (el tramo del
   2023-12-31 en test) hereda su calentamiento de la cola del train, mientras
   que un episodio sin historia previa contigua (el tramo de mayo de 2024)
   pierde sus primeras ``WARMUP_BARS`` barras como calentamiento propio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

RAW_COLUMNS = ["Timestamp", "Gmtoffset", "Datetime", "Open", "High", "Low", "Close", "Volume"]
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]


@dataclass
class AuditReport:
    """Resultado de auditar un archivo crudo de precios."""

    nombre: str
    n_filas: int
    fecha_inicio: pd.Timestamp
    fecha_fin: pd.Timestamp
    n_ohlc_vacio: int
    racha_vacio_maxima_barras: int
    n_streaks_vacio: int
    n_offgrid: int
    n_barras_planas: int
    pct_volumen_vacio: float
    huecos_grandes: list = field(default_factory=list)  # list of dicts
    gmtoffset_unico: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["fecha_inicio"] = str(self.fecha_inicio)
        d["fecha_fin"] = str(self.fecha_fin)
        return d

    def resumen_texto(self) -> str:
        lineas = [
            f"Auditoría de {self.nombre}",
            f"  Filas: {self.n_filas:,} | Rango: {self.fecha_inicio} -> {self.fecha_fin}",
            f"  Gmtoffset único: {self.gmtoffset_unico} (0 = UTC, confirmado)",
            f"  Barras con OHLC vacío: {self.n_ohlc_vacio:,} en {self.n_streaks_vacio} rachas"
            f" (racha máxima: {self.racha_vacio_maxima_barras} barras)",
            f"  Marcas fuera de rejilla de 5 min: {self.n_offgrid:,}",
            f"  Barras planas (O=H=L=C): {self.n_barras_planas:,}",
            f"  Volumen vacío: {self.pct_volumen_vacio:.1%}",
            f"  Huecos grandes de calendario (> {config.GAP_THRESHOLD_MINUTES} min): {len(self.huecos_grandes)}",
        ]
        for h in self.huecos_grandes:
            lineas.append(
                f"    - {h['inicio']} -> {h['fin']} ({h['duracion_horas']:.2f} h)"
            )
        return "\n".join(lineas)


def _load_raw_csv(path) -> pd.DataFrame:
    """Lee un CSV crudo tal cual viene, solo ordena por Timestamp."""
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def audit_dataset(df_raw: pd.DataFrame, nombre: str) -> AuditReport:
    """Audita un dataframe crudo (antes de cualquier limpieza).

    Reporta huecos de calendario, rachas de OHLC vacío, marcas fuera de
    rejilla, barras planas y proporción de volumen faltante. No modifica
    ``df_raw``.
    """
    dt = df_raw["Datetime"]
    diffs_min = dt.diff().dt.total_seconds().div(60)

    gap_mask = diffs_min > config.GAP_THRESHOLD_MINUTES
    huecos = []
    for idx in diffs_min[gap_mask].index:
        huecos.append(
            {
                "inicio": str(dt.loc[idx - 1]),
                "fin": str(dt.loc[idx]),
                "duracion_horas": float(diffs_min.loc[idx] / 60.0),
            }
        )

    is_nan = df_raw["Close"].isna()
    streak_id = (is_nan != is_nan.shift()).cumsum()
    streak_sizes = df_raw[is_nan].groupby(streak_id[is_nan]).size()

    offgrid = (dt.dt.minute % config.BAR_MINUTES) != 0

    flat = (
        (df_raw["Open"] == df_raw["High"])
        & (df_raw["High"] == df_raw["Low"])
        & (df_raw["Low"] == df_raw["Close"])
        & df_raw["Close"].notna()
    )

    return AuditReport(
        nombre=nombre,
        n_filas=len(df_raw),
        fecha_inicio=dt.iloc[0],
        fecha_fin=dt.iloc[-1],
        n_ohlc_vacio=int(is_nan.sum()),
        racha_vacio_maxima_barras=int(streak_sizes.max()) if len(streak_sizes) else 0,
        n_streaks_vacio=int(len(streak_sizes)),
        n_offgrid=int(offgrid.sum()),
        n_barras_planas=int(flat.sum()),
        pct_volumen_vacio=float(df_raw["Volume"].isna().mean()),
        huecos_grandes=huecos,
        gmtoffset_unico=sorted(df_raw["Gmtoffset"].unique().tolist()),
    )


def _assign_episodes(dt_raw: pd.Series) -> pd.Series:
    """Asigna un id de episodio a cada marca de tiempo cruda.

    Dos barras consecutivas pertenecen al mismo episodio si su diferencia de
    tiempo es menor o igual al umbral de hueco grande. Se calcula sobre las
    marcas de tiempo CRUDAS (antes de eliminar OHLC vacío) para que el límite
    de episodio dependa del calendario real y no de nuestra propia limpieza.
    """
    diffs_min = dt_raw.diff().dt.total_seconds().div(60)
    is_new_episode = diffs_min > config.GAP_THRESHOLD_MINUTES
    is_new_episode.iloc[0] = True
    return is_new_episode.cumsum().rename("episode_id")


def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Limpia un dataframe crudo de forma causal.

    Pasos:
      1. Asigna episodio a cada fila (sobre las marcas de tiempo crudas).
      2. Elimina las filas con OHLC vacío (sin relleno ni interpolación).
      3. Marca como no operable (``is_tradable=False``) cada fila dentro de
         las primeras ``config.WARMUP_BARS`` de su episodio.

    No usa ``Volume`` en ningún cálculo posterior: la columna se conserva
    solo para referencia, dado que está vacía en una proporción alta e
    irregular de las filas.
    """
    df = df_raw.copy()
    df["episode_id"] = _assign_episodes(df["Datetime"])
    df = df.dropna(subset=OHLC_COLUMNS).reset_index(drop=True)

    df["bar_in_episode"] = df.groupby("episode_id").cumcount()
    df["is_tradable"] = df["bar_in_episode"] >= config.WARMUP_BARS
    return df


def _drop_shared_boundary_row(train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> pd.DataFrame:
    """Elimina de ``test`` la barra que ya está duplicada al final de ``train``.

    La última fila de train (2023-12-31 00:00) es idéntica a la primera fila
    de test. Se conserva en train y se descarta en test para no contarla dos
    veces en los retornos ni en el conteo de operaciones.
    """
    if len(train_clean) == 0 or len(test_clean) == 0:
        return test_clean
    ultimo_train = train_clean.iloc[-1]["Timestamp"]
    primero_test = test_clean.iloc[0]["Timestamp"]
    if ultimo_train == primero_test:
        test_clean = test_clean.iloc[1:].reset_index(drop=True)
    return test_clean


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame, AuditReport, AuditReport]:
    """Carga, audita y limpia train y test.

    Devuelve ``(train_clean, test_clean, audit_train, audit_test)``. Los
    dataframes limpios incluyen las columnas ``episode_id``,
    ``bar_in_episode`` e ``is_tradable``; en ``test`` el episodio al que
    pertenece la barra 2023-12-31 sigue siendo el mismo id que el último
    episodio de train solo a nivel conceptual (no se renumeran juntos), por
    lo que el calentamiento de ese segmento se construye explícitamente con
    :func:`build_warmup_context`.
    """
    train_raw = _load_raw_csv(config.TRAIN_CSV)
    test_raw = _load_raw_csv(config.TEST_CSV)

    audit_train = audit_dataset(train_raw, "btc_project_train.csv")
    audit_test = audit_dataset(test_raw, "btc_project_test.csv")

    train_clean = clean_dataset(train_raw)
    test_clean = clean_dataset(test_raw)
    test_clean = _drop_shared_boundary_row(train_clean, test_clean)

    return train_clean, test_clean, audit_train, audit_test


def build_warmup_context(train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> pd.DataFrame:
    """Construye la serie de test con calentamiento, siguiendo la regla:

    - El primer episodio de test (2023-12-31, continuo con el final de
      train) se antepone con la cola del último episodio de train, para que
      sus indicadores y su régimen hereden historia real y no pierdan
      barras operables.
    - El resto de episodios de test (el tramo de mayo-junio de 2024, que
      sigue a un hueco de ~122 días) no recibe historia externa: calienta
      con sus propias primeras ``WARMUP_BARS`` barras, tal como ya lo marca
      ``is_tradable`` calculado en :func:`clean_dataset`.

    El resultado es un dataframe con las mismas columnas que ``test_clean``
    más una bandera ``is_test_row`` (False para las barras de train
    anexadas solo como contexto de calentamiento, True para las barras que
    sí pertenecen al archivo de test y sobre las que se reportan métricas).
    """
    test_clean = test_clean.copy()
    test_clean["is_test_row"] = True

    primer_episodio_test = test_clean["episode_id"].iloc[0]
    ultimo_episodio_train = train_clean["episode_id"].iloc[-1]

    cola_train = train_clean[train_clean["episode_id"] == ultimo_episodio_train].copy()
    cola_train = cola_train.tail(config.WARMUP_BARS)
    cola_train["is_test_row"] = False

    primer_bloque_test = test_clean[test_clean["episode_id"] == primer_episodio_test].copy()
    resto_test = test_clean[test_clean["episode_id"] != primer_episodio_test].copy()

    bloque_con_contexto = pd.concat([cola_train, primer_bloque_test], ignore_index=True)
    bloque_con_contexto["episode_id"] = primer_episodio_test
    bloque_con_contexto["bar_in_episode"] = np.arange(len(bloque_con_contexto))
    bloque_con_contexto["is_tradable"] = (
        bloque_con_contexto["bar_in_episode"] >= config.WARMUP_BARS
    ) & bloque_con_contexto["is_test_row"]

    resultado = pd.concat([bloque_con_contexto, resto_test], ignore_index=True)
    return resultado


def episode_summary(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen de episodios: inicio, fin, barras totales y operables."""
    filas = []
    for ep_id, grupo in df_clean.groupby("episode_id"):
        filas.append(
            {
                "episode_id": ep_id,
                "inicio": grupo["Datetime"].iloc[0],
                "fin": grupo["Datetime"].iloc[-1],
                "n_barras": len(grupo),
                "n_barras_operables": int(grupo["is_tradable"].sum()),
            }
        )
    return pd.DataFrame(filas)
