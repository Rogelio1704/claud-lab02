"""Funciones de graficación. Todas las figuras llevan título, ejes
etiquetados y leyenda, y se guardan como PNG en ``docs/figuras/``.

Este módulo solo dibuja: recibe series y tablas ya calculadas por
``backtest``, ``metrics``, ``regimes`` y ``optimize``, y no vuelve a correr
ninguna simulación.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna

from . import config

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def _save(fig, path):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_equity_vs_benchmark(equity: pd.Series, benchmark: pd.Series, titulo: str, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, label="Estrategia", color="#1f6feb", linewidth=1.2)
    ax.plot(benchmark.index, benchmark.values, label="Buy & Hold BTC", color="#8b949e", linewidth=1.0)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor del portafolio (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_drawdown(equity: pd.Series, titulo: str, path):
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#da3633", alpha=0.5)
    ax.plot(drawdown.index, drawdown.values, color="#da3633", linewidth=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_returns_heatmap(returns: pd.Series, titulo: str, path, date_fmt: str = "%Y-%m"):
    fig, ax = plt.subplots(figsize=(max(6, len(returns) * 0.5), 2.2))
    valores = (returns.values * 100.0).reshape(1, -1)
    im = ax.imshow(valores, cmap="RdYlGn", aspect="auto", vmin=-np.nanmax(np.abs(valores)),
                   vmax=np.nanmax(np.abs(valores)))
    ax.set_yticks([])
    ax.set_xticks(range(len(returns)))
    ax.set_xticklabels([d.strftime(date_fmt) for d in returns.index], rotation=90, fontsize=7)
    for i, v in enumerate(valores[0]):
        ax.text(i, 0, f"{v:.1f}%", ha="center", va="center", fontsize=6)
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.35, label="Retorno (%)", shrink=0.5)
    _save(fig, path)


def plot_sensitivity(sens_df: pd.DataFrame, titulo: str, path):
    datos = sens_df[sens_df["parametro"] != "ninguno"].copy()
    base_calmar = sens_df.loc[sens_df["parametro"] == "ninguno", "calmar"].iloc[0]
    datos["etiqueta"] = datos["regimen"] + " · " + datos["parametro"]
    pivot = datos.pivot_table(index="etiqueta", columns="variacion", values="calmar", aggfunc="first")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(pivot))))
    y = np.arange(len(pivot))
    width = 0.35
    cols = list(pivot.columns)
    for i, col in enumerate(cols):
        ax.barh(y + (i - 0.5) * width, pivot[col].values, height=width, label=col)
    ax.axvline(base_calmar, color="black", linestyle="--", linewidth=1, label="Calmar base")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Calmar Ratio")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    _save(fig, path)


def plot_cost_curve(cost_df: pd.DataFrame, titulo: str, path, comision_base: float = config.COMMISSION):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_df["comision"] * 100, cost_df["retorno_total"] * 100, marker="o", color="#1f6feb",
            label="Retorno neto total")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(comision_base * 100, color="#da3633", linestyle="--", label=f"Comisión base ({comision_base:.3%})")
    for mult in config.COST_MULTIPLIERS:
        ax.axvline(comision_base * 100 * mult, color="#8b949e", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Comisión por operación, entrada+salida (%)")
    ax.set_ylabel("Retorno neto total (%)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_regime_timeline(df_price: pd.DataFrame, regime_codes: pd.Series, titulo: str, path):
    colores = {0: "#2ea043", 1: "#d29922", 2: "#da3633"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    precio_grafica = df_price[["Datetime", "Close"]].copy()
    huecos = precio_grafica["Datetime"].diff() > pd.Timedelta(days=1)
    precio_grafica.loc[huecos, "Close"] = np.nan  # no conectar con una línea recta a través de un hueco grande
    ax.plot(precio_grafica["Datetime"], precio_grafica["Close"], color="black", linewidth=0.6, label="Precio BTCUSDT")
    precio_por_fecha = df_price.set_index("Datetime")["Close"]
    for code, nombre in enumerate(config.REGIME_NAMES):
        mask = regime_codes == code
        fechas = regime_codes.index[mask]
        fechas = fechas.intersection(precio_por_fecha.index)
        if len(fechas) == 0:
            continue
        ax.scatter(fechas, precio_por_fecha.loc[fechas], s=3, color=colores[code], label=nombre)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio de cierre (USD)")
    ax.legend(markerscale=3)
    _save(fig, path)


def plot_regime_variable_distributions(features: pd.DataFrame, regime_codes: pd.Series, titulo: str, path):
    cols = list(features.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.5))
    comunes = features.index.intersection(regime_codes.index)
    f = features.loc[comunes]
    r = regime_codes.loc[comunes]
    colores = {0: "#2ea043", 1: "#d29922", 2: "#da3633"}
    for ax, col in zip(axes, cols):
        for code, nombre in enumerate(config.REGIME_NAMES):
            valores = f.loc[r == code, col].dropna()
            if len(valores) > 0:
                ax.hist(valores, bins=40, alpha=0.5, density=True, label=nombre, color=colores[code])
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Densidad")
    axes[0].legend(fontsize=7)
    fig.suptitle(titulo)
    _save(fig, path)


def plot_equity_with_regimes(equity: pd.Series, regime_codes: pd.Series, titulo: str, path):
    colores = {0: "#2ea04333", 1: "#d2992233", 2: "#da363333"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(equity.index, equity.values, color="#1f6feb", linewidth=1.0, label="Valor del portafolio", zorder=3)

    regime_codes_dedup = regime_codes[~regime_codes.index.duplicated(keep="last")]
    r = regime_codes_dedup.reindex(equity.index).ffill().fillna(-1)
    valores = r.to_numpy()
    fechas = r.index.to_numpy()
    cambio = np.r_[True, valores[1:] != valores[:-1]]
    posiciones_cambio = np.nonzero(cambio)[0]
    for i, pos0 in enumerate(posiciones_cambio):
        pos1 = posiciones_cambio[i + 1] if i + 1 < len(posiciones_cambio) else len(valores) - 1
        code = int(valores[pos0])
        if code in colores:
            ax.axvspan(fechas[pos0], fechas[pos1], color=colores[code], zorder=1)

    for code, nombre in enumerate(config.REGIME_NAMES):
        ax.plot([], [], color=colores[code][:7], linewidth=8, label=nombre)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor del portafolio (USD)")
    ax.legend()
    _save(fig, path)


def plot_optimization_history(study: optuna.Study, titulo: str, path):
    df = study.trials_dataframe()
    df = df[df["value"] > config.PENALTY_VALUE / 2]
    mejor_acumulado = df["value"].cummax()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["number"], df["value"], s=10, alpha=0.5, label="Calmar por prueba", color="#8b949e")
    ax.plot(df["number"], mejor_acumulado, color="#1f6feb", linewidth=1.5, label="Mejor Calmar acumulado")
    ax.axvline(config.N_TRIALS_RANDOM, color="black", linestyle="--", linewidth=0.8,
               label="Fin de Random Search / inicio TPE")
    ax.set_xlabel("Número de prueba")
    ax.set_ylabel("Calmar Ratio")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_param_importance(study: optuna.Study, titulo: str, path):
    try:
        importancias = optuna.importance.get_param_importances(study)
    except Exception:
        importancias = {}
    if not importancias:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    nombres = list(importancias.keys())
    valores = list(importancias.values())
    ax.barh(nombres, valores, color="#1f6feb")
    ax.set_xlabel("Importancia (varianza explicada del Calmar)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3, axis="x")
    _save(fig, path)


def plot_slice(study: optuna.Study, param_names: list, titulo: str, path):
    df = study.trials_dataframe()
    df = df[df["value"] > config.PENALTY_VALUE / 2]
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
    if len(param_names) == 1:
        axes = [axes]
    for ax, p in zip(axes, param_names):
        ax.scatter(df[f"params_{p}"], df["value"], s=12, alpha=0.6, color="#1f6feb")
        ax.set_xlabel(p)
        ax.set_ylabel("Calmar Ratio")
        ax.set_title(p)
    fig.suptitle(titulo)
    _save(fig, path)


def plot_2d_surface(study: optuna.Study, param_x: str, param_y: str, titulo: str, path):
    df = study.trials_dataframe()
    df = df[df["value"] > config.PENALTY_VALUE / 2]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(df[f"params_{param_x}"], df[f"params_{param_y}"], c=df["value"], cmap="viridis", s=30)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(titulo)
    fig.colorbar(sc, ax=ax, label="Calmar Ratio")
    _save(fig, path)
