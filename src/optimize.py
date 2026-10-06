"""Optimización de hiperparámetros y walk-forward, con régimen por ventana.

Esquema híbrido "warm-start" (recomendado en clase): una fase exploratoria
corta de Random Search (``config.N_TRIALS_RANDOM`` pruebas) seguida de
Optuna con TPE (``config.N_TRIALS_TPE`` pruebas más), ambas dentro del MISMO
``optuna.Study``, de modo que el muestreador TPE use directamente las
pruebas aleatorias ya evaluadas como observaciones iniciales. El total
(``config.N_TRIALS_TOTAL`` = 120) está dentro del rango de 100 a 200 pedido
por el PDF, y se cuenta igual por régimen y por ventana.

La función objetivo (ver :func:`compute_objective`) es el Calmar Ratio del
tramo evaluado cuando el retorno anualizado es positivo, y el retorno total
de la ventana (acotado en [-1, 0]) cuando no lo es — corrección necesaria
porque el Calmar sin condicionar premia la ruina (ver su docstring).
Se penaliza con ``config.PENALTY_VALUE`` si el número de operaciones
generado es menor a ``config.N_MIN_TRADES``.

Walk-forward: ventana de entrenamiento de 1 mes, prueba de 1 semana, paso de
1 semana, sobre ``btc_project_train.csv``. En cada ventana se ajusta un
modelo de régimen causal (solo con datos hasta el final de la ventana de
entrenamiento) y se optimiza un conjunto de hiperparámetros por régimen
usando únicamente las barras de ESA ventana de entrenamiento que
pertenecen a ese régimen. Los parámetros elegidos se evalúan después, sin
reoptimizar, sobre la semana de prueba (Calmar OOS) tanto de forma aislada
por régimen (para la Walk-Forward Efficiency) como de forma conjunta y
cronológica (para la curva de equity que se reporta como resultado).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd

from . import backtest, config, metrics, regimes, signals

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Espacio de búsqueda
# ---------------------------------------------------------------------------
def suggest_params(trial: optuna.Trial) -> signals.StrategyParams:
    b = signals.PARAM_BOUNDS
    return signals.StrategyParams(
        ema_fast=trial.suggest_int("ema_fast", *b["ema_fast"]),
        ema_slow=trial.suggest_int("ema_slow", *b["ema_slow"]),
        trend_band_k=trial.suggest_float("trend_band_k", *b["trend_band_k"]),
        rsi_window=trial.suggest_int("rsi_window", *b["rsi_window"]),
        rsi_band=trial.suggest_float("rsi_band", *b["rsi_band"]),
        boll_window=trial.suggest_int("boll_window", *b["boll_window"]),
        boll_k=trial.suggest_float("boll_k", *b["boll_k"]),
        atr_window=trial.suggest_int("atr_window", *b["atr_window"]),
        sl_atr_mult=trial.suggest_float("sl_atr_mult", *b["sl_atr_mult"]),
        tp_atr_mult=trial.suggest_float("tp_atr_mult", *b["tp_atr_mult"]),
        position_fraction=trial.suggest_float("position_fraction", *b["position_fraction"]),
    )


def default_params() -> signals.StrategyParams:
    """Parámetros de respaldo (punto medio de cada rango) para la primera
    ventana en la que un régimen no tiene historia previa que heredar."""
    b = signals.PARAM_BOUNDS
    mid = {k: (lo + hi) / 2.0 for k, (lo, hi) in b.items()}
    return signals.StrategyParams(
        ema_fast=int(mid["ema_fast"]), ema_slow=int(mid["ema_slow"]),
        trend_band_k=mid["trend_band_k"], rsi_window=int(mid["rsi_window"]),
        rsi_band=mid["rsi_band"], boll_window=int(mid["boll_window"]),
        boll_k=mid["boll_k"], atr_window=int(mid["atr_window"]),
        sl_atr_mult=mid["sl_atr_mult"], tp_atr_mult=mid["tp_atr_mult"],
        position_fraction=mid["position_fraction"],
    )


# ---------------------------------------------------------------------------
# Utilidades de tramos contiguos (para aislar un régimen dentro de una ventana)
# ---------------------------------------------------------------------------
def contiguous_true_runs(mask: np.ndarray) -> list:
    """Devuelve [(inicio, fin_exclusivo), ...] de tramos consecutivos True."""
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def run_regime_subset_backtest(context_df: pd.DataFrame, held_regime: pd.Series, mask: np.ndarray,
                                params_by_regime: dict, initial_cash: float = config.INITIAL_CAPITAL,
                                end_reason: str = "fin_de_ventana", commission: float = config.COMMISSION,
                                signal_column: str = "signal") -> backtest.BacktestResult:
    """Corre el motor solo sobre las barras de ``mask`` (p. ej. un régimen
    dentro de una ventana), calculando los indicadores con TODA la historia
    causal de ``context_df`` y encadenando el capital entre tramos
    consecutivos de ``mask`` (nunca se simula sobre una barra fuera de la
    máscara, así que un cambio de régimen fuera de ella ya implica el cierre
    implícito de cualquier posición, consistente con la regla de
    transición)."""
    decision = backtest.build_decision_arrays(context_df, held_regime, params_by_regime, signal_column=signal_column)
    applied = backtest.shift_decision_to_execution(decision)

    dt = pd.DatetimeIndex(context_df["Datetime"])
    open_ = context_df["Open"].to_numpy()
    high = context_df["High"].to_numpy()
    low = context_df["Low"].to_numpy()
    close = context_df["Close"].to_numpy()
    is_tradable = context_df["is_tradable"].to_numpy()

    runs = contiguous_true_runs(mask)
    cash = float(initial_cash)
    all_trades = []
    equity_parts = []
    for start, end in runs:
        sl = slice(start, end)
        result = backtest.run_backtest(
            datetime_index=dt[sl], open_=open_[sl], high=high[sl], low=low[sl], close=close[sl],
            exec_signal=applied["exec_signal"][sl], atr=applied["atr"][sl],
            sl_mult=applied["sl_mult"][sl], tp_mult=applied["tp_mult"][sl], frac=applied["frac"][sl],
            regime=applied["regime"][sl], is_tradable=is_tradable[sl],
            commission=commission, initial_cash=cash, end_reason=end_reason,
        )
        cash = result.final_cash
        all_trades.extend(result.trades)
        equity_parts.append(result.equity)

    equity = pd.concat(equity_parts) if equity_parts else pd.Series(dtype=float)
    return backtest.BacktestResult(trades=all_trades, equity=equity, final_cash=cash, initial_cash=float(initial_cash))


# ---------------------------------------------------------------------------
# Función objetivo y búsqueda híbrida por régimen
# ---------------------------------------------------------------------------
@dataclass
class RegimeOptResult:
    regime_code: int
    params: signals.StrategyParams
    calmar_is: float
    n_trades_is: int
    n_trials: int
    selection_method: str
    inherited: bool
    trials_df: pd.DataFrame = field(repr=False, default=None)
    study: object = field(repr=False, default=None)


def compute_objective(equity: pd.Series) -> float:
    """Función objetivo corregida (ver reporte, sección de causas de desempeño).

    f(equity) = Calmar(equity)        si el retorno anualizado > 0
    f(equity) = retorno_total(equity) si el retorno anualizado <= 0   (en [-1, 0])

    Defecto corregido: el Calmar sin condicionar premia la ruina. El retorno
    anualizado tiene piso en -100% mientras que el drawdown sigue creciendo
    con la pérdida, así que Calmar = retorno_anualizado / MDD se vuelve
    MENOS negativo (mejor) cuanto más se acerca la pérdida al 100%: en una
    ventana de un mes, perder 1% da Calmar ≈ -11.4, perder 50% da ≈ -2.0 y
    perder todo da -1.0. Maximizar ese número empuja al optimizador hacia la
    ruina en vez de alejarlo de ella. Usar el retorno total (acotado en
    [-1, 0]) para las configuraciones perdedoras restaura el orden
    económicamente correcto: perder menos SIEMPRE da un valor mayor
    (más cercano a 0) que perder más, y cualquier configuración con Calmar
    positivo (por definición un número que no está en [-1, 0] salvo el caso
    degenerado) vale más que cualquier configuración perdedora.
    """
    if len(equity) < 2:
        return config.PENALTY_VALUE
    retorno_anual = metrics.annualized_return(equity)
    if retorno_anual is not None and not np.isnan(retorno_anual) and retorno_anual > 0:
        calmar = metrics.calmar_ratio(equity)
        if calmar is not None and not np.isnan(calmar):
            return float(calmar)
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def _make_objective(context_df, held_regime, is_mask, regime_code):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        result = run_regime_subset_backtest(
            context_df, held_regime, is_mask, {regime_code: params}, end_reason="fin_de_ventana_is",
        )
        trades = result.trades_frame()
        n_trades = len(trades)
        trial.set_user_attr("n_trades", n_trades)
        if n_trades < config.N_MIN_TRADES:
            return config.PENALTY_VALUE
        return compute_objective(result.equity)

    return objective


def select_plateau_params(trials_df: pd.DataFrame, n_bins: int = 4) -> tuple:
    """Elige θ* evitando un pico aislado: agrupa pruebas válidas en una
    rejilla gruesa del espacio de parámetros y prefiere la celda con mejor
    Calmar PROMEDIO entre las que tienen al menos 2 pruebas (una meseta). Si
    ninguna celda tiene vecinos, cae de vuelta al argmax simple y lo declara.
    """
    valid = trials_df[trials_df["value"] > config.PENALTY_VALUE / 2].copy()
    if valid.empty:
        return None, {"method": "sin_configuracion_valida", "n_valid_trials": 0}

    param_names = list(signals.PARAM_BOUNDS.keys())
    for p in param_names:
        lo, hi = signals.PARAM_BOUNDS[p]
        col = f"params_{p}"
        norm = (valid[col] - lo) / (hi - lo + 1e-12)
        valid[f"_bin_{p}"] = np.clip((norm * n_bins).astype(int), 0, n_bins - 1)

    bin_cols = [f"_bin_{p}" for p in param_names]
    valid["_cell"] = valid[bin_cols].astype(str).agg("-".join, axis=1)

    cell_stats = valid.groupby("_cell")["value"].agg(["mean", "count"])
    candidatas = cell_stats[cell_stats["count"] >= 2]

    if candidatas.empty:
        best_row = valid.loc[valid["value"].idxmax()]
        info = {
            "method": "argmax_pico_aislado",
            "n_valid_trials": len(valid),
            "n_celdas_candidatas": 0,
            "chosen_trial_number": int(best_row["number"]),
            "chosen_value": float(best_row["value"]),
        }
    else:
        best_cell = candidatas["mean"].idxmax()
        en_celda = valid[valid["_cell"] == best_cell]
        mediana = en_celda["value"].median()
        representante = en_celda.iloc[(en_celda["value"] - mediana).abs().argsort().iloc[0]]
        info = {
            "method": "meseta",
            "n_valid_trials": len(valid),
            "n_celdas_candidatas": int(len(candidatas)),
            "mejor_celda_media": float(candidatas.loc[best_cell, "mean"]),
            "mejor_celda_n": int(candidatas.loc[best_cell, "count"]),
            "chosen_trial_number": int(representante["number"]),
            "chosen_value": float(representante["value"]),
        }
        best_row = representante

    params_dict = {p: best_row[f"params_{p}"] for p in param_names}
    for intcol in ("ema_fast", "ema_slow", "rsi_window", "boll_window", "atr_window"):
        params_dict[intcol] = int(params_dict[intcol])
    params = signals.StrategyParams(**params_dict)
    return params, info


def optimize_regime_window(context_df, held_regime, is_mask, regime_code, seed: int,
                            previous_params=None) -> RegimeOptResult:
    """Corre el esquema híbrido Random->TPE para un régimen dentro de una
    ventana de entrenamiento y elige theta* con la regla de meseta."""
    n_rows_regime = int(is_mask.sum())
    min_rows_heuristic = config.N_MIN_TRADES * 5
    if n_rows_regime < min_rows_heuristic:
        params = previous_params if previous_params is not None else default_params()
        return RegimeOptResult(
            regime_code=regime_code, params=params, calmar_is=np.nan, n_trades_is=0,
            n_trials=0, selection_method="heredado_pocas_barras", inherited=True, trials_df=None,
        )

    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = _make_objective(context_df, held_regime, is_mask, regime_code)
    study.optimize(objective, n_trials=config.N_TRIALS_RANDOM, show_progress_bar=False)

    study.sampler = optuna.samplers.TPESampler(seed=seed + 1, multivariate=True, n_startup_trials=0)
    study.optimize(objective, n_trials=config.N_TRIALS_TPE, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    params, info = select_plateau_params(trials_df)

    if params is None:
        params = previous_params if previous_params is not None else default_params()
        return RegimeOptResult(
            regime_code=regime_code, params=params, calmar_is=np.nan, n_trades_is=0,
            n_trials=len(trials_df), selection_method="heredado_sin_validos", inherited=True,
            trials_df=trials_df, study=study,
        )

    result = run_regime_subset_backtest(context_df, held_regime, is_mask, {regime_code: params},
                                         end_reason="fin_de_ventana_is")
    trades = result.trades_frame()
    calmar_is = metrics.calmar_ratio(result.equity)

    return RegimeOptResult(
        regime_code=regime_code, params=params, calmar_is=calmar_is, n_trades_is=len(trades),
        n_trials=len(trials_df), selection_method=info["method"], inherited=False, trials_df=trials_df,
        study=study,
    )


# ---------------------------------------------------------------------------
# Corrida de referencia SIN separación por régimen (pregunta de análisis 5)
# ---------------------------------------------------------------------------
def _make_objective_baseline(context_df, held_dummy, is_mask):
    """Como `_make_objective`, pero con UN solo conjunto de parámetros
    aplicado a toda la ventana (sin distinguir régimen): se logra
    reutilizando el motor con los tres códigos de régimen apuntando al
    MISMO objeto `StrategyParams`, así que `build_decision_arrays` calcula
    los indicadores una sola vez y los aplica a todas las barras por igual,
    exactamente como si no existiera la capa de régimen."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        params_por_regimen = {r: params for r in range(config.N_REGIMES)}
        result = run_regime_subset_backtest(
            context_df, held_dummy, is_mask, params_por_regimen, end_reason="fin_de_ventana_is",
        )
        trades = result.trades_frame()
        n_trades = len(trades)
        trial.set_user_attr("n_trades", n_trades)
        if n_trades < config.N_MIN_TRADES:
            return config.PENALTY_VALUE
        return compute_objective(result.equity)

    return objective


def optimize_baseline_window(context_df, held_dummy, is_mask, seed: int,
                              previous_params=None) -> RegimeOptResult:
    """Optimiza UN solo conjunto de hiperparámetros para toda la ventana de
    entrenamiento, sin separar por régimen. Mismo esquema híbrido
    Random->TPE, mismo presupuesto de pruebas (``config.N_TRIALS_TOTAL``),
    misma función objetivo corregida y la misma restricción de operaciones
    mínimas que la versión con régimen (`optimize_regime_window`): la única
    diferencia de diseño es que aquí no hay modelo de régimen que ajustar.
    Se usa como comparación para la pregunta de análisis 5 (¿qué aporta la
    capa de régimen?).
    """
    n_rows = int(is_mask.sum())
    min_rows_heuristic = config.N_MIN_TRADES * 5
    if n_rows < min_rows_heuristic:
        params = previous_params if previous_params is not None else default_params()
        return RegimeOptResult(
            regime_code=-1, params=params, calmar_is=np.nan, n_trades_is=0,
            n_trials=0, selection_method="heredado_pocas_barras", inherited=True, trials_df=None,
        )

    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = _make_objective_baseline(context_df, held_dummy, is_mask)
    study.optimize(objective, n_trials=config.N_TRIALS_RANDOM, show_progress_bar=False)

    study.sampler = optuna.samplers.TPESampler(seed=seed + 1, multivariate=True, n_startup_trials=0)
    study.optimize(objective, n_trials=config.N_TRIALS_TPE, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    params, info = select_plateau_params(trials_df)

    if params is None:
        params = previous_params if previous_params is not None else default_params()
        return RegimeOptResult(
            regime_code=-1, params=params, calmar_is=np.nan, n_trades_is=0,
            n_trials=len(trials_df), selection_method="heredado_sin_validos", inherited=True,
            trials_df=trials_df, study=study,
        )

    params_por_regimen = {r: params for r in range(config.N_REGIMES)}
    result = run_regime_subset_backtest(context_df, held_dummy, is_mask, params_por_regimen,
                                         end_reason="fin_de_ventana_is")
    trades = result.trades_frame()
    calmar_is = metrics.calmar_ratio(result.equity)

    return RegimeOptResult(
        regime_code=-1, params=params, calmar_is=calmar_is, n_trades_is=len(trades),
        n_trials=len(trials_df), selection_method=info["method"], inherited=False, trials_df=trials_df,
        study=study,
    )


# ---------------------------------------------------------------------------
# Ventanas de walk-forward
# ---------------------------------------------------------------------------
def generate_walkforward_windows(df_clean: pd.DataFrame) -> list:
    """Genera (is_start, is_end, oos_start, oos_end) en calendario, recortados
    para no cruzar un hueco grande dentro de la semana de prueba y sin
    empezar antes del inicio del episodio al que pertenece esa semana."""
    start = df_clean["Datetime"].iloc[0]
    end = df_clean["Datetime"].iloc[-1]
    episodio_inicio = df_clean.groupby("episode_id")["Datetime"].min()

    windows = []
    cursor = start
    while True:
        is_start = cursor
        is_end = is_start + pd.Timedelta(days=config.WALKFORWARD_TRAIN_DAYS)
        oos_start = is_end
        oos_end = oos_start + pd.Timedelta(days=config.WALKFORWARD_TEST_DAYS)
        if oos_end > end:
            break

        oos_rows = df_clean[(df_clean["Datetime"] >= oos_start) & (df_clean["Datetime"] < oos_end)]
        cursor = cursor + pd.Timedelta(days=config.WALKFORWARD_STEP_DAYS)
        if oos_rows.empty:
            continue
        if oos_rows["episode_id"].nunique() > 1:
            continue  # un hueco grande cae dentro de la semana de prueba: se descarta

        ep_id = oos_rows["episode_id"].iloc[0]
        ep_start = episodio_inicio.loc[ep_id]
        is_start_clip = max(is_start, ep_start)
        if is_start_clip >= is_end:
            continue

        windows.append((is_start_clip, is_end, oos_start, oos_end, int(ep_id)))
    return windows


def build_context_for_window(df_clean: pd.DataFrame, is_start, is_end, oos_end, ep_id: int) -> pd.DataFrame:
    """Recorta ``df_clean`` a un contexto causal suficiente: desde (el mayor
    entre el inicio del episodio y) ``is_start`` menos el margen de la
    ventana de régimen, hasta ``oos_end``."""
    ep_start = df_clean.loc[df_clean["episode_id"] == ep_id, "Datetime"].min()
    margen = pd.Timedelta(minutes=config.REGIME_WINDOW_BARS * config.BAR_MINUTES)
    context_start = max(ep_start, is_start - margen)
    mask = (df_clean["Datetime"] >= context_start) & (df_clean["Datetime"] < oos_end) & (df_clean["episode_id"] == ep_id)
    return df_clean.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward completo
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    windows_table: pd.DataFrame
    oos_equity: pd.Series
    oos_trades: pd.DataFrame
    regime_codes_oos: pd.Series
    regime_model_last: object
    params_by_regime_last: dict
    elapsed_seconds: float
    n_configs_evaluated: int
    silhouette_by_window: pd.DataFrame
    window_cache: list = field(default_factory=list)
    last_window_studies: dict = field(default_factory=dict)


def run_walkforward(train_clean: pd.DataFrame, seed: int = config.RANDOM_SEED,
                     max_windows: int | None = None, verbose: bool = True) -> WalkForwardResult:
    t0 = time.time()
    windows = generate_walkforward_windows(train_clean)
    if max_windows is not None:
        windows = windows[:max_windows]

    params_by_regime = {r: default_params() for r in range(config.N_REGIMES)}
    rows = []
    oos_equity_parts = []
    oos_trades_parts = []
    oos_regime_parts = []
    silhouette_rows = []
    window_cache = []
    n_configs = 0
    cash_joint = config.INITIAL_CAPITAL

    for w_idx, (is_start, is_end, oos_start, oos_end, ep_id) in enumerate(windows):
        context_df = build_context_for_window(train_clean, is_start, is_end, oos_end, ep_id)
        features = regimes.compute_regime_features(context_df)

        is_feat_mask = (context_df["Datetime"] >= is_start) & (context_df["Datetime"] < is_end)
        fit_rows = features.loc[is_feat_mask.to_numpy()].dropna()
        if len(fit_rows) < config.N_REGIMES * 10:
            continue

        model = regimes.fit_regime_model(fit_rows, seed=seed + w_idx)
        codes = model.predict_codes(features)
        held = regimes.labels_with_update_cadence(codes)

        is_mask = ((context_df["Datetime"] >= is_start) & (context_df["Datetime"] < is_end)
                   & context_df["is_tradable"]).to_numpy()
        oos_mask = ((context_df["Datetime"] >= oos_start) & (context_df["Datetime"] < oos_end)
                    & context_df["is_tradable"]).to_numpy()

        window_params = {}
        window_studies = {}
        for r in range(config.N_REGIMES):
            regime_is_mask = is_mask & (held.to_numpy() == r)
            opt_result = optimize_regime_window(
                context_df, held, regime_is_mask, r, seed=seed + w_idx * 10 + r,
                previous_params=params_by_regime.get(r),
            )
            window_params[r] = opt_result.params
            window_studies[r] = opt_result.study
            n_configs += opt_result.n_trials

            regime_oos_mask = oos_mask & (held.to_numpy() == r)
            oos_result = run_regime_subset_backtest(
                context_df, held, regime_oos_mask, {r: opt_result.params}, end_reason="fin_de_ventana_oos",
            )
            n_trades_oos = len(oos_result.trades_frame())
            calmar_oos = metrics.calmar_ratio(oos_result.equity) if n_trades_oos > 0 else np.nan

            wfe = np.nan
            if not np.isnan(opt_result.calmar_is) and not np.isnan(calmar_oos) and opt_result.calmar_is > 0.05:
                wfe = calmar_oos / opt_result.calmar_is

            rows.append({
                "window": w_idx, "is_start": is_start, "is_end": is_end,
                "oos_start": oos_start, "oos_end": oos_end, "regime": config.REGIME_NAMES[r],
                "regime_code": r, "n_trials": opt_result.n_trials,
                "selection_method": opt_result.selection_method, "inherited": opt_result.inherited,
                "n_trades_is": opt_result.n_trades_is, "calmar_is": opt_result.calmar_is,
                "n_trades_oos": n_trades_oos, "calmar_oos": calmar_oos, "wfe": wfe,
                **{f"param_{k}": v for k, v in opt_result.params.as_dict().items()},
            })

            if opt_result.params is not None:
                params_by_regime[r] = opt_result.params

        silhouette_rows.append({"window": w_idx, "silhouette": model.silhouette, "n_fit_rows": model.n_fit_rows})

        oos_slice_mask = ((context_df["Datetime"] >= oos_start) & (context_df["Datetime"] < oos_end)).to_numpy()
        joint_result = run_regime_subset_backtest(
            context_df, held, oos_slice_mask, window_params, initial_cash=cash_joint,
            end_reason="fin_de_ventana_oos",
        )
        cash_joint = joint_result.final_cash
        oos_equity_parts.append(joint_result.equity)
        if len(joint_result.trades) > 0:
            oos_trades_parts.append(joint_result.trades_frame())
        held_oos_por_fecha = held.loc[oos_slice_mask].set_axis(
            pd.DatetimeIndex(context_df.loc[oos_slice_mask, "Datetime"])
        )
        oos_regime_parts.append(held_oos_por_fecha)

        window_cache.append({
            "window": w_idx, "context_df": context_df, "held": held,
            "params_by_regime": window_params, "oos_mask": oos_slice_mask,
        })

        regime_model_last = model
        params_by_regime_last = window_params
        last_window_studies = window_studies

        if verbose:
            print(f"  ventana {w_idx+1}/{len(windows)}: {is_start.date()} -> {oos_end.date()} "
                  f"| configs acumuladas={n_configs}")

    elapsed = time.time() - t0
    oos_equity = pd.concat(oos_equity_parts) if oos_equity_parts else pd.Series(dtype=float)
    oos_trades = pd.concat(oos_trades_parts, ignore_index=True) if oos_trades_parts else pd.DataFrame()
    oos_regime = pd.concat(oos_regime_parts) if oos_regime_parts else pd.Series(dtype=int)

    return WalkForwardResult(
        windows_table=pd.DataFrame(rows), oos_equity=oos_equity, oos_trades=oos_trades,
        regime_codes_oos=oos_regime, regime_model_last=regime_model_last,
        params_by_regime_last=params_by_regime_last, elapsed_seconds=elapsed,
        n_configs_evaluated=n_configs, silhouette_by_window=pd.DataFrame(silhouette_rows),
        window_cache=window_cache, last_window_studies=last_window_studies,
    )


@dataclass
class BaselineWalkForwardResult:
    windows_table: pd.DataFrame
    oos_equity: pd.Series
    oos_trades: pd.DataFrame
    elapsed_seconds: float
    n_configs_evaluated: int


def run_walkforward_baseline(train_clean: pd.DataFrame, seed: int = config.RANDOM_SEED,
                              max_windows: int | None = None, verbose: bool = True) -> BaselineWalkForwardResult:
    """Walk-forward de referencia SIN separación por régimen (pregunta de
    análisis 5): mismas ventanas (`generate_walkforward_windows`), mismo
    presupuesto de pruebas por ventana, misma función objetivo corregida y
    los mismos costos que `run_walkforward`. No ajusta ningún modelo de
    régimen (no hace falta: usa un código de régimen "dummy" constante, que
    nunca puede disparar un cierre por cambio de régimen)."""
    t0 = time.time()
    windows = generate_walkforward_windows(train_clean)
    if max_windows is not None:
        windows = windows[:max_windows]

    previous_params = default_params()
    rows = []
    oos_equity_parts = []
    oos_trades_parts = []
    n_configs = 0
    cash_joint = config.INITIAL_CAPITAL

    for w_idx, (is_start, is_end, oos_start, oos_end, ep_id) in enumerate(windows):
        context_df = build_context_for_window(train_clean, is_start, is_end, oos_end, ep_id)
        held_dummy = pd.Series(0, index=context_df.index, dtype=int)

        is_mask = ((context_df["Datetime"] >= is_start) & (context_df["Datetime"] < is_end)
                   & context_df["is_tradable"]).to_numpy()
        oos_mask = ((context_df["Datetime"] >= oos_start) & (context_df["Datetime"] < oos_end)
                    & context_df["is_tradable"]).to_numpy()

        opt_result = optimize_baseline_window(context_df, held_dummy, is_mask, seed=seed + w_idx,
                                               previous_params=previous_params)
        previous_params = opt_result.params
        n_configs += opt_result.n_trials

        params_por_regimen = {r: opt_result.params for r in range(config.N_REGIMES)}
        oos_result = run_regime_subset_backtest(context_df, held_dummy, oos_mask, params_por_regimen,
                                                  initial_cash=cash_joint, end_reason="fin_de_ventana_oos")
        cash_joint = oos_result.final_cash
        n_trades_oos = len(oos_result.trades_frame())
        calmar_oos = metrics.calmar_ratio(oos_result.equity) if n_trades_oos > 0 else np.nan

        wfe = np.nan
        if not np.isnan(opt_result.calmar_is) and not np.isnan(calmar_oos) and opt_result.calmar_is > 0.05:
            wfe = calmar_oos / opt_result.calmar_is

        rows.append({
            "window": w_idx, "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
            "n_trials": opt_result.n_trials, "selection_method": opt_result.selection_method,
            "inherited": opt_result.inherited, "n_trades_is": opt_result.n_trades_is,
            "calmar_is": opt_result.calmar_is, "n_trades_oos": n_trades_oos, "calmar_oos": calmar_oos,
            "wfe": wfe, **{f"param_{k}": v for k, v in opt_result.params.as_dict().items()},
        })

        oos_equity_parts.append(oos_result.equity)
        if len(oos_result.trades) > 0:
            oos_trades_parts.append(oos_result.trades_frame())

        if verbose:
            print(f"  [sin regimen] ventana {w_idx+1}/{len(windows)}: {is_start.date()} -> {oos_end.date()} "
                  f"| configs acumuladas={n_configs}")

    elapsed = time.time() - t0
    oos_equity = pd.concat(oos_equity_parts) if oos_equity_parts else pd.Series(dtype=float)
    oos_trades = pd.concat(oos_trades_parts, ignore_index=True) if oos_trades_parts else pd.DataFrame()

    return BaselineWalkForwardResult(
        windows_table=pd.DataFrame(rows), oos_equity=oos_equity, oos_trades=oos_trades,
        elapsed_seconds=elapsed, n_configs_evaluated=n_configs,
    )


# ---------------------------------------------------------------------------
# Re-simulación rápida del tramo OOS de train (sin volver a ajustar régimen
# ni a optimizar): se usa para sensibilidad de parámetros y barrido de costos.
# ---------------------------------------------------------------------------
def resimulate_oos_path(window_cache: list, params_override: dict = None,
                         commission: float = config.COMMISSION,
                         initial_cash: float = config.INITIAL_CAPITAL,
                         signal_column: str = "signal") -> backtest.BacktestResult:
    """Vuelve a correr la trayectoria OOS completa del walk-forward de train.

    Reutiliza, para cada ventana, el mismo contexto y la misma etiqueta de
    régimen ya calculados (``window_cache``), así que nunca vuelve a ajustar
    el modelo de régimen ni a optimizar. Si ``params_override`` es ``None``
    usa los parámetros que esa ventana eligió originalmente; si se da un
    diccionario ``{regime_code: StrategyParams}``, lo aplica POR IGUAL en
    todas las ventanas (para medir sensibilidad del set final congelado a lo
    largo de toda la trayectoria OOS, no solo en la última semana).
    """
    cash = float(initial_cash)
    all_trades = []
    equity_parts = []
    for item in window_cache:
        params = params_override if params_override is not None else item["params_by_regime"]
        result = run_regime_subset_backtest(
            item["context_df"], item["held"], item["oos_mask"], params,
            initial_cash=cash, end_reason="fin_de_ventana_oos",
            commission=commission, signal_column=signal_column,
        )
        cash = result.final_cash
        equity_parts.append(result.equity)
        all_trades.extend(result.trades)

    equity = pd.concat(equity_parts) if equity_parts else pd.Series(dtype=float)
    return backtest.BacktestResult(trades=all_trades, equity=equity, final_cash=cash, initial_cash=float(initial_cash))


def sensitivity_analysis(window_cache: list, frozen_params_by_regime: dict,
                          pct: float = config.SENSITIVITY_PCT) -> pd.DataFrame:
    """Sensibilidad ±20%: perturba un hiperparámetro a la vez, en los
    parámetros finales congelados, y reevalúa sobre toda la trayectoria OOS
    de train (ver :func:`resimulate_oos_path`). Reporta Calmar, retorno
    total Y número de operaciones de cada variante (antes solo se reportaba
    el Calmar, que con el defecto 1.1 ya corregido podía dar -1.000 en
    bloque sin informar si eso venía de pocas operaciones o de muchas)."""
    base_result = resimulate_oos_path(window_cache, params_override=frozen_params_by_regime)
    base_equity = base_result.equity
    base_calmar = metrics.calmar_ratio(base_equity)

    filas = [{
        "regimen": "base", "parametro": "ninguno", "variacion": 0.0,
        "calmar": base_calmar, "retorno_total": base_equity.iloc[-1] / base_equity.iloc[0] - 1.0,
        "n_operaciones": len(base_result.trades),
    }]

    param_names = list(signals.PARAM_BOUNDS.keys())
    for r, params in frozen_params_by_regime.items():
        base_dict = params.as_dict()
        for p in param_names:
            for signo, etiqueta in [(1 + pct, f"+{int(pct*100)}%"), (1 - pct, f"-{int(pct*100)}%")]:
                nuevo_dict = base_dict.copy()
                nuevo_dict[p] = base_dict[p] * signo
                if p in ("ema_fast", "ema_slow", "rsi_window", "boll_window", "atr_window"):
                    nuevo_dict[p] = max(1, int(round(nuevo_dict[p])))
                nuevo_params = signals.StrategyParams(**nuevo_dict)
                override = dict(frozen_params_by_regime)
                override[r] = nuevo_params
                result = resimulate_oos_path(window_cache, params_override=override)
                equity = result.equity
                calmar = metrics.calmar_ratio(equity)
                retorno = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 1 else np.nan
                filas.append({
                    "regimen": config.REGIME_NAMES[r], "parametro": p, "variacion": etiqueta,
                    "calmar": calmar, "retorno_total": retorno, "n_operaciones": len(result.trades),
                })
    return pd.DataFrame(filas)


def cost_sensitivity_sweep(window_cache: list, frozen_params_by_regime: dict,
                            commission_levels: np.ndarray) -> pd.DataFrame:
    """Retorno neto y Calmar sobre la trayectoria OOS de train, barriendo el
    nivel de comisión desde 0 hasta bien pasado el punto de equilibrio."""
    filas = []
    for c in commission_levels:
        result = resimulate_oos_path(window_cache, params_override=frozen_params_by_regime, commission=c)
        equity = result.equity
        retorno = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 1 else np.nan
        calmar = metrics.calmar_ratio(equity)
        filas.append({"comision": c, "retorno_total": retorno, "calmar": calmar, "n_operaciones": len(result.trades)})
    return pd.DataFrame(filas)


def freeze_final_config(wf_result: "WalkForwardResult") -> dict:
    """Serializa el modelo de régimen y los parámetros por régimen de la
    última ventana de entrenamiento, para escribirlos en
    ``docs/tablas/`` ANTES de tocar el test (ver `main.py`)."""
    model = wf_result.regime_model_last
    congelado = {
        "regime_model": {
            "scaler_mean": model.scaler.mean_.tolist(),
            "scaler_scale": model.scaler.scale_.tolist(),
            "kmeans_centers": model.kmeans.cluster_centers_.tolist(),
            "cluster_to_code": {int(k): int(v) for k, v in model.cluster_to_code.items()},
            "silhouette_ultima_ventana": model.silhouette,
            "feature_columns": regimes.FEATURE_COLUMNS,
        },
        "params_by_regime": {
            config.REGIME_NAMES[r]: p.as_dict() for r, p in wf_result.params_by_regime_last.items()
        },
        "n_configs_evaluadas_walkforward": wf_result.n_configs_evaluated,
        "tiempo_optimizacion_segundos": wf_result.elapsed_seconds,
    }
    return congelado


def evaluate_on_test(train_clean: pd.DataFrame, test_context: pd.DataFrame, regime_model,
                      params_by_regime: dict) -> dict:
    """Evalúa, una sola vez, los parámetros y el modelo de régimen congelados
    sobre el test. ``test_context`` es el resultado de
    :func:`src.data.build_warmup_context`: incluye la cola de train pegada
    al primer episodio de test (para calentar indicadores y régimen) y
    marca con ``is_test_row`` las barras que sí pertenecen al test.

    Devuelve un diccionario con el resultado conjunto (ambos segmentos
    encadenados) y el resultado de cada segmento por separado.
    """
    resultados_segmento = {}
    cash = config.INITIAL_CAPITAL
    equity_parts = []
    all_trades = []
    regime_parts = []

    for ep_id, sub in test_context.groupby("episode_id"):
        sub = sub.reset_index(drop=True)
        features = regimes.compute_regime_features(sub)
        codes = regime_model.predict_codes(features)
        held = regimes.labels_with_update_cadence(codes)
        mask = sub["is_test_row"].to_numpy()
        if not mask.any():
            continue

        result = run_regime_subset_backtest(
            sub, held, mask, params_by_regime, initial_cash=cash, end_reason="fin_de_test",
        )
        cash = result.final_cash
        equity_parts.append(result.equity)
        all_trades.extend(result.trades)
        regime_parts.append(held.loc[mask].set_axis(pd.DatetimeIndex(sub.loc[mask, "Datetime"])))

        resultados_segmento[int(ep_id)] = {
            "result": result,
            "fecha_inicio": sub.loc[mask, "Datetime"].iloc[0],
            "fecha_fin": sub.loc[mask, "Datetime"].iloc[-1],
            "regime_codes": held.loc[mask],
        }

    equity_total = pd.concat(equity_parts) if equity_parts else pd.Series(dtype=float)
    regime_total = pd.concat(regime_parts) if regime_parts else pd.Series(dtype=int)
    combinado = backtest.BacktestResult(trades=all_trades, equity=equity_total,
                                         final_cash=cash, initial_cash=config.INITIAL_CAPITAL)

    return {"combinado": combinado, "por_segmento": resultados_segmento, "regimen_total": regime_total}


def single_indicator_comparison(window_cache: list, frozen_params_by_regime: dict) -> pd.DataFrame:
    """Compara la regla 2 de 3 contra usar cada indicador por separado,
    sobre la misma trayectoria OOS de train y con los mismos parámetros
    finales congelados (se cambia solo la columna de señal usada)."""
    filas = []
    for nombre, columna in [("confirmacion_2_de_3", "signal"), ("solo_tendencia", "dir_trend"),
                             ("solo_momento", "dir_momentum"), ("solo_volatilidad", "dir_vol")]:
        result = resimulate_oos_path(window_cache, params_override=frozen_params_by_regime,
                                      signal_column=columna)
        equity = result.equity
        calmar = metrics.calmar_ratio(equity)
        filas.append({
            "variante": nombre, "n_operaciones": len(result.trades), "calmar": calmar,
            "retorno_total": equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 1 else np.nan,
        })
    return pd.DataFrame(filas)
