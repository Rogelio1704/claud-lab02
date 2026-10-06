"""Motor de backtesting event-driven, barra por barra.

Convenciones y supuestos (declarados también en el reporte):

- Sin apalancamiento: tanto en largos como en cortos, el capital
  comprometido es ``qty * precio_entrada`` y nunca puede superar el efectivo
  disponible en ese momento. Para los cortos se reserva el 100% del nocional
  como colateral (convención conservadora, simétrica con los largos); no se
  modela costo de préstamo ni funding.
- Comisión de ``config.COMMISSION`` sobre el nocional, tanto en la apertura
  como en el cierre de cada operación, sin excepción (stop-loss,
  take-profit, cambio de régimen, hueco de datos o fin de periodo pagan la
  misma comisión de cierre).
- Si el stop-loss y el take-profit caen dentro del rango [Low, High] de la
  misma barra, se ejecuta primero el stop-loss (convención conservadora).
- Si la barra abre más allá del nivel de stop-loss o take-profit de una
  posición que ya estaba abierta, se ejecuta al precio de apertura de esa
  barra (no al precio teórico del nivel), porque ese precio ya no está
  disponible.
- Una posición nunca cruza un hueco grande de datos: el llamador nunca debe
  pasar a ``run_backtest`` un tramo que contenga uno (ver `src/data.py`,
  episodios). El motor, por eso, siempre fuerza el cierre de cualquier
  posición abierta en la última barra de la serie que recibe.
- Toda la lógica de selección de parámetros por régimen (qué ventana de
  indicador, qué multiplicador de SL/TP, qué fracción de posición aplica en
  cada barra) ocurre ANTES de llamar a este módulo. `run_backtest` solo
  conoce arreglos ya resueltos barra a barra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, signals


def build_decision_arrays(df_episode: pd.DataFrame, regime_series: pd.Series,
                           params_by_regime: dict, signal_column: str = "signal") -> dict:
    """Arma, para cada barra, la decisión tomada con los parámetros de SU régimen.

    Para cada régimen ``r`` con hiperparámetros ``params_by_regime[r]`` se
    calcula la tabla de indicadores completa sobre ``df_episode`` (causal,
    con toda la historia disponible del episodio). Luego, barra por barra,
    se selecciona el valor que corresponde al régimen vigente en ESA barra
    (``regime_series``). El resultado todavía es la "decisión tomada al
    cierre de la barra t"; el desplazamiento a "ejecución en t+1" ocurre en
    :func:`shift_decision_to_execution`.

    ``signal_column`` permite reutilizar esta misma función para el análisis
    de robustez de la regla de confirmación: con ``"dir_trend"``,
    ``"dir_momentum"`` o ``"dir_vol"`` en vez de ``"signal"`` se simula la
    estrategia usando un solo indicador en lugar de la confirmación 2 de 3.
    """
    n = len(df_episode)
    regime_arr = regime_series.to_numpy()
    signal_dec = np.zeros(n, dtype=float)
    atr_dec = np.full(n, np.nan, dtype=float)
    sl_dec = np.zeros(n, dtype=float)
    tp_dec = np.zeros(n, dtype=float)
    frac_dec = np.zeros(n, dtype=float)

    for r, params in params_by_regime.items():
        mask = regime_arr == r
        if not mask.any():
            continue
        table = signals.compute_indicator_table(df_episode, params)
        signal_dec[mask] = table[signal_column].to_numpy()[mask]
        atr_dec[mask] = table["atr"].to_numpy()[mask]
        sl_dec[mask] = params.sl_atr_mult
        tp_dec[mask] = params.tp_atr_mult
        frac_dec[mask] = params.position_fraction

    return {
        "signal": signal_dec,
        "atr": atr_dec,
        "sl_mult": sl_dec,
        "tp_mult": tp_dec,
        "frac": frac_dec,
        "regime": regime_arr,
    }


def shift_decision_to_execution(decision: dict) -> dict:
    """Desplaza todas las series de decisión una barra para ejecutarse en t+1.

    El régimen NO se desplaza: se usa tal cual para detectar, barra a barra,
    si el régimen vigente acaba de cambiar respecto de la barra anterior.
    """
    def _shift(arr, fill=0.0):
        out = np.empty_like(arr)
        out[0] = fill
        out[1:] = arr[:-1]
        return out

    return {
        "exec_signal": _shift(decision["signal"], 0.0).astype(int),
        "atr": _shift(decision["atr"], np.nan),
        "sl_mult": _shift(decision["sl_mult"], 0.0),
        "tp_mult": _shift(decision["tp_mult"], 0.0),
        "frac": _shift(decision["frac"], 0.0),
        "regime": decision["regime"],
    }


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int  # +1 largo, -1 corto
    qty: float
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    commission_entry: float
    commission_exit: float
    pnl: float
    regime_entry: int


@dataclass
class BacktestResult:
    trades: list
    equity: pd.Series
    final_cash: float
    initial_cash: float

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "entry_idx", "exit_idx", "entry_time", "exit_time", "side", "qty",
                    "entry_price", "exit_price", "stop_loss", "take_profit", "exit_reason",
                    "commission_entry", "commission_exit", "pnl", "regime_entry",
                ]
            )
        return pd.DataFrame([t.__dict__ for t in self.trades])


def run_backtest(
    datetime_index: pd.DatetimeIndex,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    exec_signal: np.ndarray,
    atr: np.ndarray,
    sl_mult: np.ndarray,
    tp_mult: np.ndarray,
    frac: np.ndarray,
    regime: np.ndarray,
    is_tradable: np.ndarray,
    commission: float = config.COMMISSION,
    initial_cash: float = config.INITIAL_CAPITAL,
    end_reason: str = "fin_de_periodo",
) -> BacktestResult:
    """Simula la estrategia barra por barra sobre un tramo contiguo.

    Todos los arreglos deben estar alineados y ya desplazados correctamente:
    ``exec_signal[i]``, ``atr[i]``, ``sl_mult[i]``, ``tp_mult[i]`` y
    ``frac[i]`` son los valores DECIDIDOS al cierre de la barra i-1 que se
    EJECUTAN en la apertura de la barra i (ver `signals.shift_signal_for_execution`
    y `backtest.select_regime_aware_series`). El motor no vuelve a desplazar
    nada: solo consume lo que recibe.
    """
    n = len(open_)
    cash = float(initial_cash)
    equity = np.empty(n, dtype=float)

    position_open = False
    side = 0
    qty = 0.0
    entry_price = 0.0
    entry_idx = -1
    stop_loss = 0.0
    take_profit = 0.0
    regime_entry = -1

    trades: list[Trade] = []

    for i in range(n):
        just_opened_this_bar = False

        # A) Cierre forzado por cambio de régimen (posición abierta en una barra
        # con régimen distinto al de la barra anterior).
        if position_open and i > 0 and regime[i] != regime[i - 1]:
            exit_price = float(open_[i])
            cash = _settle_exit(cash, side, qty, entry_price, exit_price, commission)
            trades.append(_make_trade(
                entry_idx, i, datetime_index[entry_idx], datetime_index[i], side, qty,
                entry_price, exit_price, stop_loss, take_profit, "cambio_regimen",
                commission, entry_price, exit_price, regime_entry,
            ))
            position_open = False

        # B) Cierre forzado por señal contraria.
        if position_open and exec_signal[i] != 0 and exec_signal[i] != side:
            exit_price = float(open_[i])
            cash = _settle_exit(cash, side, qty, entry_price, exit_price, commission)
            trades.append(_make_trade(
                entry_idx, i, datetime_index[entry_idx], datetime_index[i], side, qty,
                entry_price, exit_price, stop_loss, take_profit, "senal_contraria",
                commission, entry_price, exit_price, regime_entry,
            ))
            position_open = False

        # C) Nueva entrada (solo si está plano, la barra es operable y hay señal).
        if not position_open and is_tradable[i] and exec_signal[i] != 0 and atr[i] > 0 and not np.isnan(atr[i]):
            side = int(exec_signal[i])
            entry_price = float(open_[i])
            qty = (frac[i] * cash) / (entry_price * (1.0 + commission))
            if qty > 0:
                notional = qty * entry_price
                cash -= notional + notional * commission
                if side == 1:
                    stop_loss = entry_price - sl_mult[i] * atr[i]
                    take_profit = entry_price + tp_mult[i] * atr[i]
                else:
                    stop_loss = entry_price + sl_mult[i] * atr[i]
                    take_profit = entry_price - tp_mult[i] * atr[i]
                entry_idx = i
                regime_entry = int(regime[i])
                position_open = True
                just_opened_this_bar = True

        # D) Revisión de stop-loss / take-profit contra High/Low de esta barra.
        if position_open:
            exit_price = None
            reason = None
            if side == 1:
                sl_hit = low[i] <= stop_loss
                tp_hit = high[i] >= take_profit
                if not just_opened_this_bar and open_[i] <= stop_loss:
                    exit_price, reason = float(open_[i]), "stop_loss"
                elif not just_opened_this_bar and open_[i] >= take_profit:
                    exit_price, reason = float(open_[i]), "take_profit"
                elif sl_hit:
                    exit_price, reason = float(stop_loss), "stop_loss"
                elif tp_hit:
                    exit_price, reason = float(take_profit), "take_profit"
            else:
                sl_hit = high[i] >= stop_loss
                tp_hit = low[i] <= take_profit
                if not just_opened_this_bar and open_[i] >= stop_loss:
                    exit_price, reason = float(open_[i]), "stop_loss"
                elif not just_opened_this_bar and open_[i] <= take_profit:
                    exit_price, reason = float(open_[i]), "take_profit"
                elif sl_hit:
                    exit_price, reason = float(stop_loss), "stop_loss"
                elif tp_hit:
                    exit_price, reason = float(take_profit), "take_profit"

            if exit_price is not None:
                cash = _settle_exit(cash, side, qty, entry_price, exit_price, commission)
                trades.append(_make_trade(
                    entry_idx, i, datetime_index[entry_idx], datetime_index[i], side, qty,
                    entry_price, exit_price, stop_loss, take_profit, reason,
                    commission, entry_price, exit_price, regime_entry,
                ))
                position_open = False

        # E) Marca a mercado del valor del portafolio al cierre de la barra i.
        if position_open:
            if side == 1:
                equity[i] = cash + qty * entry_price + qty * (close[i] - entry_price)
            else:
                notional = qty * entry_price
                equity[i] = cash + notional + qty * (entry_price - close[i])
        else:
            equity[i] = cash

    # Cierre forzado al final de la serie (fin de ventana, episodio o huecograde).
    if position_open:
        i = n - 1
        exit_price = float(close[i])
        cash = _settle_exit(cash, side, qty, entry_price, exit_price, commission)
        trades.append(_make_trade(
            entry_idx, i, datetime_index[entry_idx], datetime_index[i], side, qty,
            entry_price, exit_price, stop_loss, take_profit, end_reason,
            commission, entry_price, exit_price, regime_entry,
        ))
        equity[i] = cash

    equity_series = pd.Series(equity, index=datetime_index, name="equity")
    return BacktestResult(trades=trades, equity=equity_series, final_cash=cash, initial_cash=float(initial_cash))


def _settle_exit(cash: float, side: int, qty: float, entry_price: float, exit_price: float, commission: float) -> float:
    if side == 1:
        proceeds = qty * exit_price
        return cash + proceeds - proceeds * commission
    else:
        notional_entry = qty * entry_price
        exit_notional = qty * exit_price
        commission_exit = exit_notional * commission
        return cash + notional_entry + qty * (entry_price - exit_price) - commission_exit


def _make_trade(entry_idx, exit_idx, entry_time, exit_time, side, qty, entry_price, exit_price,
                 stop_loss, take_profit, reason, commission, entry_notional_price, exit_notional_price,
                 regime_entry) -> Trade:
    commission_entry = qty * entry_price * commission
    commission_exit = qty * exit_price * commission
    if side == 1:
        pnl = qty * (exit_price - entry_price) - commission_entry - commission_exit
    else:
        pnl = qty * (entry_price - exit_price) - commission_entry - commission_exit
    return Trade(
        entry_idx=entry_idx, exit_idx=exit_idx, entry_time=entry_time, exit_time=exit_time,
        side=side, qty=qty, entry_price=entry_price, exit_price=exit_price,
        stop_loss=stop_loss, take_profit=take_profit, exit_reason=reason,
        commission_entry=commission_entry, commission_exit=commission_exit, pnl=pnl,
        regime_entry=regime_entry,
    )


def run_backtest_on_episode(
    df_episode: pd.DataFrame,
    regime_series: pd.Series,
    params_by_regime: dict,
    commission: float = config.COMMISSION,
    initial_cash: float = config.INITIAL_CAPITAL,
    end_reason: str = "fin_de_periodo",
    signal_column: str = "signal",
) -> BacktestResult:
    """Arma las series de decisión por régimen, las desplaza y corre el motor.

    ``df_episode`` debe tener columnas Open, High, Low, Close, Datetime,
    is_tradable y estar ordenado cronológicamente sin huecos grandes
    internos (un episodio, o un tramo contiguo de un mismo régimen dentro de
    un episodio).
    """
    decision = build_decision_arrays(df_episode, regime_series, params_by_regime, signal_column=signal_column)
    applied = shift_decision_to_execution(decision)

    return run_backtest(
        datetime_index=pd.DatetimeIndex(df_episode["Datetime"]),
        open_=df_episode["Open"].to_numpy(),
        high=df_episode["High"].to_numpy(),
        low=df_episode["Low"].to_numpy(),
        close=df_episode["Close"].to_numpy(),
        exec_signal=applied["exec_signal"],
        atr=applied["atr"],
        sl_mult=applied["sl_mult"],
        tp_mult=applied["tp_mult"],
        frac=applied["frac"],
        regime=applied["regime"],
        is_tradable=df_episode["is_tradable"].to_numpy(),
        commission=commission,
        initial_cash=initial_cash,
        end_reason=end_reason,
    )


def run_backtest_chained(
    segments: list,
    regime_series_by_segment: list,
    params_by_regime: dict,
    commission: float = config.COMMISSION,
    initial_cash: float = config.INITIAL_CAPITAL,
    end_reason: str = "fin_de_periodo",
) -> BacktestResult:
    """Corre varios tramos contiguos encadenando el capital entre ellos.

    Se usa tanto para unir tramos de un mismo régimen que no son contiguos en
    el tiempo (optimización por régimen) como para unir episodios separados
    por un hueco grande (por ejemplo, los dos segmentos del test). El
    capital final de un tramo es el capital inicial del siguiente; ningún
    tramo individual cruza un hueco.
    """
    cash = float(initial_cash)
    all_trades: list[Trade] = []
    equity_parts = []
    for seg_df, seg_regime in zip(segments, regime_series_by_segment):
        if len(seg_df) == 0:
            continue
        result = run_backtest_on_episode(
            seg_df, seg_regime, params_by_regime, commission=commission,
            initial_cash=cash, end_reason=end_reason,
        )
        cash = result.final_cash
        all_trades.extend(result.trades)
        equity_parts.append(result.equity)

    if equity_parts:
        equity = pd.concat(equity_parts)
    else:
        equity = pd.Series(dtype=float)

    return BacktestResult(trades=all_trades, equity=equity, final_cash=cash, initial_cash=float(initial_cash))


def portfolio_value_check(result: BacktestResult) -> dict:
    """Prueba de contabilidad: efectivo + valor de posiciones abiertas == equity final.

    Como `run_backtest` siempre fuerza el cierre de cualquier posición al
    final de la serie, al terminar no debe quedar valor en posiciones
    abiertas: ``final_cash`` debe coincidir exactamente con la última barra
    de la curva de equity.
    """
    last_equity = float(result.equity.iloc[-1]) if len(result.equity) else result.initial_cash
    return {
        "final_cash": result.final_cash,
        "last_equity": last_equity,
        "coincide": np.isclose(result.final_cash, last_equity, rtol=1e-9, atol=1e-6),
    }


def total_commissions_check(result: BacktestResult, commission: float = config.COMMISSION) -> dict:
    """Prueba de contabilidad: comisiones pagadas == operaciones * comisión * nocional."""
    trades = result.trades_frame()
    if trades.empty:
        return {"comisiones_registradas": 0.0, "comisiones_esperadas": 0.0, "coincide": True}
    esperado = (
        trades["qty"] * trades["entry_price"] * commission
        + trades["qty"] * trades["exit_price"] * commission
    ).sum()
    registrado = (trades["commission_entry"] + trades["commission_exit"]).sum()
    return {
        "comisiones_registradas": float(registrado),
        "comisiones_esperadas": float(esperado),
        "coincide": bool(np.isclose(registrado, esperado, rtol=1e-9, atol=1e-6)),
    }
