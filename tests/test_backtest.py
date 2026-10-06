"""Pruebas del motor de backtesting: contabilidad, SL antes que TP, ausencia
de apalancamiento, cierre antes de un hueco y un golden-file test con un
resultado calculado a mano."""

import numpy as np
import pandas as pd
import pytest

from src import backtest, config


def _arrays(dt, open_, high, low, close, exec_signal, atr, sl_mult, tp_mult, frac, regime=None, is_tradable=None):
    n = len(open_)
    if regime is None:
        regime = np.zeros(n, dtype=int)
    if is_tradable is None:
        is_tradable = np.ones(n, dtype=bool)
    return dict(
        datetime_index=pd.DatetimeIndex(dt), open_=np.array(open_, dtype=float),
        high=np.array(high, dtype=float), low=np.array(low, dtype=float), close=np.array(close, dtype=float),
        exec_signal=np.array(exec_signal, dtype=int), atr=np.array(atr, dtype=float),
        sl_mult=np.array(sl_mult, dtype=float), tp_mult=np.array(tp_mult, dtype=float),
        frac=np.array(frac, dtype=float), regime=np.array(regime, dtype=int),
        is_tradable=np.array(is_tradable, dtype=bool),
    )


def test_golden_file_largo_con_take_profit():
    """Caso calculado a mano: una entrada larga que cierra por take-profit.

    Entrada en t=1 a precio 100, frac=0.5, comisión=0.001 (para que el
    cálculo a mano sea simple). ATR=2, TP_mult=3 -> take_profit=106.
    qty = 0.5 * 1_000_000 / (100 * 1.001) = 4995.00999...
    En t=2 el precio alcanza High=110 (>=106): sale en 106 por take_profit.
    pnl = qty*(106-100) - qty*100*0.001 - qty*106*0.001
    """
    dt = pd.date_range("2024-01-01", periods=3, freq="5min")
    comision = 0.001
    res = backtest.run_backtest(
        **_arrays(
            dt, open_=[100, 100, 106], high=[100, 101, 110], low=[100, 99, 105], close=[100, 100, 108],
            exec_signal=[0, 1, 0], atr=[np.nan, 2.0, 2.0], sl_mult=[0, 2.0, 2.0], tp_mult=[0, 3.0, 3.0],
            frac=[0, 0.5, 0.5],
        ),
        commission=comision, initial_cash=1_000_000.0,
    )
    trades = res.trades_frame()
    assert len(trades) == 1
    qty_esperada = 0.5 * 1_000_000.0 / (100 * (1 + comision))
    pnl_esperado = qty_esperada * (106 - 100) - qty_esperada * 100 * comision - qty_esperada * 106 * comision
    assert trades.iloc[0]["exit_reason"] == "take_profit"
    assert np.isclose(trades.iloc[0]["qty"], qty_esperada, rtol=1e-9)
    assert np.isclose(trades.iloc[0]["pnl"], pnl_esperado, rtol=1e-9)
    assert np.isclose(res.final_cash, 1_000_000.0 + pnl_esperado, rtol=1e-9)


def test_stop_loss_se_ejecuta_antes_que_take_profit_en_la_misma_barra():
    dt = pd.date_range("2024-01-01", periods=2, freq="5min")
    res = backtest.run_backtest(
        **_arrays(
            dt, open_=[100, 100], high=[100, 200], low=[100, 50], close=[100, 150],
            exec_signal=[0, 1], atr=[np.nan, 2.0], sl_mult=[0, 2.0], tp_mult=[0, 3.0], frac=[0, 0.5],
        ),
        commission=config.COMMISSION, initial_cash=1_000_000.0,
    )
    trades = res.trades_frame()
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "stop_loss"


def test_sin_apalancamiento_nocional_nunca_excede_efectivo_disponible():
    dt = pd.date_range("2024-01-01", periods=6, freq="5min")
    res = backtest.run_backtest(
        **_arrays(
            dt, open_=[100, 100, 100, 100, 100, 100], high=[100, 101, 101, 101, 101, 101],
            low=[100, 99, 99, 99, 99, 99], close=[100, 100.5, 100.2, 100.6, 100.1, 100.3],
            exec_signal=[0, 1, -1, 1, -1, 0], atr=[np.nan, 5.0, 5.0, 5.0, 5.0, 5.0],
            sl_mult=[0, 2.0, 2.0, 2.0, 2.0, 2.0], tp_mult=[0, 10.0, 10.0, 10.0, 10.0, 10.0],
            frac=[0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        commission=config.COMMISSION, initial_cash=1_000_000.0,
    )
    trades = res.trades_frame()
    cash = 1_000_000.0
    for _, t in trades.iterrows():
        notional = t["qty"] * t["entry_price"]
        assert notional * (1 + config.COMMISSION) <= cash + 1e-6
        cash += t["pnl"]


def test_posicion_se_cierra_antes_de_un_hueco_grande_al_encadenar_tramos():
    """Dos tramos separados por un hueco grande deben simularse por separado,
    encadenando el capital: la posición abierta al final del primer tramo se
    fuerza a cerrar ahí, nunca sobrevive al segundo tramo."""
    dt1 = pd.date_range("2024-01-01", periods=2, freq="5min")
    dt2 = pd.date_range("2024-01-05", periods=2, freq="5min")  # separado por un hueco grande

    seg1 = _arrays(
        dt1, open_=[100, 100], high=[100, 101], low=[100, 99], close=[100, 100.5],
        exec_signal=[0, 1], atr=[np.nan, 5.0], sl_mult=[0, 2.0], tp_mult=[0, 10.0], frac=[0, 0.5],
    )
    seg2 = _arrays(
        dt2, open_=[80, 80], high=[80, 81], low=[79, 79], close=[80, 80.2],
        exec_signal=[0, 0], atr=[np.nan, 5.0], sl_mult=[0, 2.0], tp_mult=[0, 10.0], frac=[0, 0.5],
    )

    res1 = backtest.run_backtest(**seg1, commission=config.COMMISSION, initial_cash=1_000_000.0,
                                  end_reason="hueco_de_datos")
    assert len(res1.trades) == 1
    assert res1.trades[0].exit_reason == "hueco_de_datos"
    assert res1.trades[0].exit_idx == 1  # se cerró en la última barra del tramo, no siguió abierta

    res2 = backtest.run_backtest(**seg2, commission=config.COMMISSION, initial_cash=res1.final_cash)
    assert len(res2.trades) == 0  # no hay señal en el segundo tramo: nunca hereda la posición anterior


def test_contabilidad_cash_mas_posiciones_abiertas_igual_equity():
    dt = pd.date_range("2024-01-01", periods=4, freq="5min")
    res = backtest.run_backtest(
        **_arrays(
            dt, open_=[100, 100, 101, 102], high=[100, 102, 103, 103], low=[100, 99, 100, 101],
            close=[100, 101, 102, 102.5], exec_signal=[0, 1, 0, 0], atr=[np.nan, 3.0, 3.0, 3.0],
            sl_mult=[0, 5.0, 5.0, 5.0], tp_mult=[0, 5.0, 5.0, 5.0], frac=[0, 0.4, 0.4, 0.4],
        ),
        commission=config.COMMISSION, initial_cash=1_000_000.0,
    )
    check = backtest.portfolio_value_check(res)
    assert check["coincide"]


def test_cierre_por_cambio_de_regimen_no_se_adelanta_una_barra():
    """Defecto 1.3: la etiqueta de régimen en la barra t se calcula con el
    cierre de t, así que un cambio de régimen detectado ahí solo puede
    forzar un cierre en la APERTURA de t+1 (misma convención que la señal),
    nunca en la apertura de t. Esta prueba habría fallado con la versión
    anterior de `shift_decision_to_execution`, que no desplazaba el
    régimen: esa versión cerraba una barra antes de lo debido."""
    n = 6
    decision = {
        "signal": np.array([0, 1, 0, 0, 0, 0], dtype=float),
        "atr": np.full(n, 5.0),
        "sl_mult": np.full(n, 2.0),
        "tp_mult": np.full(n, 20.0),  # TP muy lejos: solo el cambio de régimen puede cerrar
        "frac": np.full(n, 0.3),
        "regime": np.array([0, 0, 0, 1, 1, 1]),  # el régimen pasa a 1 al CIERRE de la barra 2
    }
    applied = backtest.shift_decision_to_execution(decision)

    dt = pd.date_range("2024-01-01", periods=n, freq="5min")
    res = backtest.run_backtest(
        datetime_index=pd.DatetimeIndex(dt),
        open_=np.full(n, 100.0), high=np.full(n, 101.0), low=np.full(n, 99.0), close=np.full(n, 100.0),
        exec_signal=applied["exec_signal"], atr=applied["atr"], sl_mult=applied["sl_mult"],
        tp_mult=applied["tp_mult"], frac=applied["frac"], regime=applied["regime"],
        is_tradable=np.ones(n, dtype=bool), commission=0.001, initial_cash=1_000_000.0,
    )
    trades = res.trades_frame()
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "cambio_regimen"
    assert trades.iloc[0]["entry_idx"] == 2
    # el cambio se "conoce" al cierre de la barra 2 (índice 3 del arreglo
    # crudo de régimen); con el desplazamiento correcto el cierre forzado
    # ejecuta en la apertura de la barra 4, no de la 3.
    assert trades.iloc[0]["exit_idx"] == 4


def test_comisiones_totales_coinciden_con_operaciones_por_comision():
    dt = pd.date_range("2024-01-01", periods=5, freq="5min")
    res = backtest.run_backtest(
        **_arrays(
            dt, open_=[100, 100, 101, 95, 95], high=[100, 102, 103, 96, 96],
            low=[100, 99, 100, 94, 94], close=[100, 101, 95, 95.5, 95.2],
            exec_signal=[0, 1, 0, 0, 0], atr=[np.nan, 3.0, 3.0, 3.0, 3.0],
            sl_mult=[0, 1.0, 1.0, 1.0, 1.0], tp_mult=[0, 10.0, 10.0, 10.0, 10.0], frac=[0, 0.3, 0.3, 0.3, 0.3],
        ),
        commission=config.COMMISSION, initial_cash=1_000_000.0,
    )
    check = backtest.total_commissions_check(res)
    assert check["coincide"]
