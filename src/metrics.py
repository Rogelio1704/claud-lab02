"""Métricas de desempeño y tablas de retornos.

Convenciones declaradas (ver también el reporte):

- BTCUSDT opera 24/7 los 365 días del año, así que el factor de
  anualización de Sharpe y Sortino usa ``config.ANNUALIZATION_FACTOR_BARS``
  (barras de 5 minutos por año), no los ~252 días hábiles de las acciones.
- Sharpe y Sortino se calculan sobre los retornos simples barra a barra
  (5 minutos) de la curva de equity.
- La tasa libre de riesgo se asume ``config.RISK_FREE_RATE_ANNUAL`` (0% anual
  declarado): a la escala de 5 minutos su efecto es numéricamente
  despreciable frente a la volatilidad de BTC y no cambia ninguna
  conclusión del proyecto.
- El retorno anualizado (y, por lo tanto, el Calmar) se calcula con el
  tiempo CALENDARIO realmente transcurrido entre la primera y la última
  barra (``365 * dias/dias_totales``), no con el conteo de barras. Esto
  evita que una ventana corta o una ventana con un hueco grande (donde
  pasan días sin que haya barras) infle o distorsione la anualización.
- Si el drawdown máximo es 0 (nunca hubo una caída desde un máximo), el
  Calmar se reporta como ``NaN`` con una nota, en vez de un número
  arbitrariamente grande sin significado económico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def bar_returns(equity: pd.Series) -> pd.Series:
    """Retornos simples barra a barra de una curva de equity."""
    return equity.pct_change().dropna()


def annualized_return(equity: pd.Series) -> float:
    """CAGR usando el tiempo calendario transcurrido (365 días/año)."""
    if len(equity) < 2:
        return np.nan
    dias = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    if dias <= 0:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0]
    if total_return <= 0:
        return -1.0
    return total_return ** (365.0 / dias) - 1.0


def annualized_volatility(equity: pd.Series) -> float:
    r = bar_returns(equity)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(config.ANNUALIZATION_FACTOR_BARS))


def sharpe_ratio(equity: pd.Series, rf_annual: float = config.RISK_FREE_RATE_ANNUAL) -> float:
    r = bar_returns(equity)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    rf_bar = (1.0 + rf_annual) ** (1.0 / config.ANNUALIZATION_FACTOR_BARS) - 1.0
    exceso = r - rf_bar
    return float(exceso.mean() / r.std(ddof=1) * np.sqrt(config.ANNUALIZATION_FACTOR_BARS))


def sortino_ratio(equity: pd.Series, rf_annual: float = config.RISK_FREE_RATE_ANNUAL) -> float:
    r = bar_returns(equity)
    if len(r) < 2:
        return np.nan
    rf_bar = (1.0 + rf_annual) ** (1.0 / config.ANNUALIZATION_FACTOR_BARS) - 1.0
    exceso = r - rf_bar
    downside = exceso[exceso < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    if downside_std == 0:
        return np.nan
    return float(exceso.mean() / downside_std * np.sqrt(config.ANNUALIZATION_FACTOR_BARS))


def max_drawdown(equity: pd.Series) -> float:
    """Magnitud (positiva) de la peor caída desde un máximo acumulado."""
    if len(equity) < 2:
        return np.nan
    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    return float(-drawdown.min())


def calmar_ratio(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    cagr = annualized_return(equity)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(cagr / mdd)


def atr_cost_diagnostic(df: pd.DataFrame, atr_window: int = 16,
                         commission: float = config.COMMISSION) -> dict:
    """Relación entre el costo de ida y vuelta y el ATR (corrección 1.2).

    Se calcula sobre datos reales de ``df`` (se usa train): el ATR(``atr_window``)
    como fracción del precio, y el múltiplo de ATR que un take-profit
    necesitaría para apenas cubrir el costo de entrar y salir
    (``2 * commission``). Motiva el rediseño de ``signals.PARAM_BOUNDS``
    para ``sl_atr_mult``/``tp_atr_mult``: con los rangos anteriores, la
    mayoría del espacio de búsqueda generaba take-profits que, aun
    acertando, no cubrían el costo de la operación.
    """
    from . import signals  # import local: evita un ciclo de importación

    atr_series = signals.atr(df["High"], df["Low"], df["Close"], atr_window)
    ratio = (atr_series / df["Close"]).dropna()
    costo_ida_vuelta = 2.0 * commission
    mediana = float(ratio.median())
    return {
        "atr_window": atr_window,
        "costo_ida_vuelta": costo_ida_vuelta,
        "atr_frac_precio_mediana": mediana,
        "atr_frac_precio_p25": float(ratio.quantile(0.25)),
        "atr_frac_precio_p75": float(ratio.quantile(0.75)),
        "tp_mult_equilibrio_mediana": costo_ida_vuelta / mediana,
        "tp_mult_equilibrio_p25": costo_ida_vuelta / float(ratio.quantile(0.25)),
        "tp_mult_equilibrio_p75": costo_ida_vuelta / float(ratio.quantile(0.75)),
    }


def trade_returns_pct(trades: pd.DataFrame) -> pd.Series:
    """Retorno porcentual por operación: PnL entre el nocional de entrada.

    Corrección (ver reporte, sección de causas): comparar el PnL en DÓLARES
    entre regímenes mezcla el efecto del régimen con el tamaño del capital
    en el momento en que ocurrió cada operación (que cambia mucho a lo
    largo de la trayectoria OOS). El retorno porcentual por operación
    (``pnl / (qty * entry_price)``) es comparable entre operaciones sin
    importar cuánto capital hubiera en ese momento.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    notional = trades["qty"] * trades["entry_price"]
    return (trades["pnl"] / notional).rename("retorno_pct")


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    return float((trades["pnl"] > 0).mean())


def payoff_ratio(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    ganancias = trades.loc[trades["pnl"] > 0, "pnl"]
    perdidas = trades.loc[trades["pnl"] < 0, "pnl"]
    if len(ganancias) == 0 or len(perdidas) == 0:
        return np.nan
    return float(ganancias.mean() / abs(perdidas.mean()))


def benchmark_equity_chained(price_df: pd.DataFrame, initial_cash: float = config.INITIAL_CAPITAL,
                              commission: float = config.COMMISSION) -> pd.Series:
    """Curva de comprar y mantener BTC, encadenada por episodios.

    Corrección (ver reporte, sección de causas): la versión anterior
    compraba una sola vez al inicio y marcaba a mercado hasta el final, sin
    importar si en medio había un hueco grande de calendario. Para el
    benchmark de test eso significaba cruzar gratis el hueco de ~122 días
    (con su salto de +38%) que la ESTRATEGIA tiene prohibido cruzar, y
    tampoco pagaba comisión de salida: una comparación injusta.

    Aquí ``price_df`` se parte en episodios y el capital se encadena entre
    ellos: se "compra" al inicio de cada uno (comisión de entrada) y se
    "vende" al final (comisión de salida), igual que la estrategia nunca
    sostiene una posición a través de un hueco grande.

    Si ``price_df`` ya trae una columna ``episode_id`` (el caso normal: se
    propaga desde ``src/data.py``, calculada sobre las marcas de tiempo
    CRUDAS de todo el archivo) se usa esa columna directamente. Es
    importante NO volver a derivar episodios con
    ``data._assign_episodes`` sobre ``price_df`` cuando este ya pasó por la
    limpieza causal (sin relleno de OHLC vacío): las rachas de barras
    eliminadas dejan huecos de calendario entre las filas que SÍ quedan, y
    recalcular episodios ahí los confundiría con huecos grandes reales
    (justo la distinción que ``data.py`` ya resuelve correctamente sobre
    los datos crudos). Solo si no hay ``episode_id`` disponible se recurre,
    como respaldo, a derivar episodios por diferencia de tiempo.
    """
    from . import data  # import local: evita un ciclo de importación con data.py

    price_df = price_df.reset_index(drop=True)
    if "episode_id" in price_df.columns:
        episodios = price_df["episode_id"].to_numpy()
    else:
        episodios = data._assign_episodes(price_df["Datetime"]).to_numpy()

    cash = float(initial_cash)
    partes = []
    for _, grupo in price_df.groupby(episodios):
        close = grupo["Close"]
        unidades = (cash * (1.0 - commission)) / close.iloc[0]
        curva = unidades * close
        cash = float(curva.iloc[-1] * (1.0 - commission))
        partes.append(pd.Series(curva.to_numpy(), index=pd.DatetimeIndex(grupo["Datetime"]), name="benchmark"))
    return pd.concat(partes)


def summary_table(equity: pd.Series, trades: pd.DataFrame, label: str,
                   benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Tabla con las métricas obligatorias más los complementos pedidos."""
    fila = {
        "conjunto": label,
        "sharpe": sharpe_ratio(equity),
        "sortino": sortino_ratio(equity),
        "calmar": calmar_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(trades),
        "retorno_anualizado": annualized_return(equity),
        "volatilidad_anualizada": annualized_volatility(equity),
        "n_operaciones": int(len(trades)),
        "payoff_ratio": payoff_ratio(trades),
        "retorno_total": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else np.nan,
    }
    if benchmark is not None and len(benchmark) > 1:
        fila["benchmark_retorno_total"] = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
        fila["benchmark_calmar"] = calmar_ratio(benchmark)
        fila["benchmark_max_drawdown"] = max_drawdown(benchmark)
    return pd.DataFrame([fila])


def periodic_returns_table(equity: pd.Series, freq: str) -> pd.Series:
    """Retornos por periodo (``'ME'``, ``'QE'`` o ``'YE'``) a partir del valor de equity."""
    period_end_values = equity.resample(freq).last().dropna()
    period_start_values = equity.resample(freq).first().dropna()
    returns = (period_end_values / period_start_values - 1.0).rename(f"retorno_{freq}")
    return returns


def monthly_quarterly_annual_tables(equity: pd.Series) -> dict:
    return {
        "mensual": periodic_returns_table(equity, "ME"),
        "trimestral": periodic_returns_table(equity, "QE"),
        "anual": periodic_returns_table(equity, "YE"),
    }


def roll_spread_estimate(close: pd.Series) -> float:
    """Medio-spread implícito del modelo de Roll (1984), como fracción del precio.

    Roll(1984): spread = 2*sqrt(-Cov(ΔP_t, ΔP_{t-1})) cuando esa covarianza
    serial es negativa (el supuesto de rebote del modelo). Si la covarianza
    observada es positiva o cero, el modelo no es aplicable en esa muestra y
    se reporta 0.0 con la advertencia correspondiente (se documenta en el
    reporte, no se fuerza un número sin sustento).
    """
    delta = close.diff().dropna()
    if len(delta) < 3:
        return 0.0
    cov = float(np.cov(delta.iloc[:-1], delta.iloc[1:])[0, 1])
    if cov >= 0:
        return 0.0
    spread = 2.0 * np.sqrt(-cov)
    return float((spread / 2.0) / close.mean())


def market_impact_fraction(sigma_bar: float, order_notional: float, daily_dollar_volume: float,
                            y: float) -> float:
    """Impacto de mercado como fracción del precio, ley de la raíz cuadrada:

    ΔP/P = Y · σ · sqrt(V_orden / V_diario)

    ``sigma_bar`` es la volatilidad de los retornos a la frecuencia de la
    barra (5 min); ``daily_dollar_volume`` y ``y`` son supuestos externos
    sin fuente propia (ver ``estimate_execution_friction_grid``).
    """
    if daily_dollar_volume <= 0:
        return 0.0
    return float(y * sigma_bar * np.sqrt(order_notional / daily_dollar_volume))


def estimate_execution_friction_grid(trades: pd.DataFrame, close: pd.Series,
                                      y_range: tuple = config.MARKET_IMPACT_Y_RANGE,
                                      volume_range: tuple = config.ASSUMED_DAILY_DOLLAR_VOLUME_RANGE,
                                      n_steps: int = 3) -> tuple:
    """Estima la magnitud de la fricción de ejecución no modelada por el backtest.

    C_trade = C_comisión + Spread/2 + ΔP_impacto + C_financiamiento

    El backtest ya cobra C_comisión. Aquí se estima Spread/2 (modelo de
    Roll, sobre los datos reales) e impacto de mercado (ley de la raíz
    cuadrada) por operación. ``Y`` y el volumen diario NO salen de los
    datos ni de la clase: son supuestos externos sin fuente propia, así que
    en vez de fijar un punto se barre una rejilla de ``n_steps × n_steps``
    combinaciones dentro de ``y_range`` y ``volume_range``, y se reporta el
    rango resultante de costo y PnL recalculado, no una cifra única. El
    financiamiento de los cortos se discute solo de forma cualitativa (no
    se modela, por la convención de simetría declarada en el motor).

    Devuelve ``(tabla, resumen)``: ``tabla`` tiene una fila por combinación
    de (Y, volumen); ``resumen`` trae el spread, el PnL original y los
    extremos (mínimo y máximo) del costo de fricción y del PnL recalculado
    sobre toda la rejilla.
    """
    half_spread_frac = roll_spread_estimate(close)
    sigma_bar = float(close.pct_change().std(ddof=0))
    pnl_original = float(trades["pnl"].sum()) if not trades.empty else 0.0
    notional = trades["qty"] * trades["entry_price"] if not trades.empty else pd.Series(dtype=float)

    y_valores = np.linspace(y_range[0], y_range[1], n_steps)
    vol_valores = np.linspace(volume_range[0], volume_range[1], n_steps)

    filas = []
    for y in y_valores:
        for vol in vol_valores:
            if trades.empty:
                costo_total, impacto_prom = 0.0, 0.0
            else:
                impacto = notional.apply(lambda v: market_impact_fraction(sigma_bar, v, vol, y))
                friccion_por_lado = half_spread_frac + impacto
                costo_total = float((2.0 * friccion_por_lado * notional).sum())
                impacto_prom = float(impacto.mean())
            filas.append({
                "Y": float(y), "volumen_diario_usd": float(vol),
                "impacto_promedio_frac": impacto_prom, "costo_friccion_total": costo_total,
                "pnl_con_friccion": pnl_original - costo_total,
            })
    tabla = pd.DataFrame(filas)

    resumen = {
        "half_spread_frac": half_spread_frac, "sigma_bar": sigma_bar,
        "pnl_original": pnl_original, "n_operaciones": len(trades),
        "costo_friccion_min": float(tabla["costo_friccion_total"].min()),
        "costo_friccion_max": float(tabla["costo_friccion_total"].max()),
        "pnl_con_friccion_min": float(tabla["pnl_con_friccion"].min()),
        "pnl_con_friccion_max": float(tabla["pnl_con_friccion"].max()),
    }
    return tabla, resumen


def bootstrap_difference_test(sample_a: pd.Series, sample_b: pd.Series, n_boot: int = 2000,
                               seed: int = config.RANDOM_SEED) -> dict:
    """Prueba de bootstrap sobre la diferencia de medias de dos muestras de retornos.

    Se usa para contrastar si el desempeño (retornos por operación o por
    barra) difiere de forma significativa entre dos regímenes. Devuelve la
    diferencia observada, el intervalo de confianza al 95% por percentiles y
    si ese intervalo excluye el cero.
    """
    rng = np.random.default_rng(seed)
    a = sample_a.dropna().to_numpy()
    b = sample_b.dropna().to_numpy()
    if len(a) < 5 or len(b) < 5:
        return {"diferencia_observada": np.nan, "ci_95": (np.nan, np.nan), "significativo": False,
                "n_a": len(a), "n_b": len(b)}
    obs_diff = a.mean() - b.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        boot_a = rng.choice(a, size=len(a), replace=True)
        boot_b = rng.choice(b, size=len(b), replace=True)
        diffs[i] = boot_a.mean() - boot_b.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diferencia_observada": float(obs_diff),
        "ci_95": (float(lo), float(hi)),
        "significativo": bool(lo > 0 or hi < 0),
        "n_a": len(a),
        "n_b": len(b),
    }
