"""Punto de entrada único del proyecto: ``python main.py``.

Ejecuta, en orden, todo el pipeline del Laboratorio 02 (Nivel B):

1. Carga, audita y limpia los datos de train y test.
2. Corre el walk-forward sobre train (optimización híbrida Random->TPE por
   régimen, 1 mes de entrenamiento / 1 semana de prueba / paso semanal).
3. Congela los parámetros finales y el modelo de régimen en
   ``docs/tablas/parametros_congelados.json`` ANTES de tocar el test.
4. Evalúa, una sola vez, el test completo y cada uno de sus dos segmentos.
5. Calcula métricas, tablas de retornos y validación de régimen para train y
   test.
6. Corre el análisis de robustez (sensibilidad ±20%, barrido de costos,
   comparación 2 de 3 contra un solo indicador, fricción de ejecución).
7. Genera las figuras y tablas obligatorias en ``docs/figuras`` y
   ``docs/tablas``.

No recibe argumentos. El tiempo aproximado de ejecución y la semilla usada
se documentan en el README.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import pandas as pd

from src import config, data, metrics, optimize, plots, regimes


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def seccion(titulo: str) -> None:
    print("\n" + "=" * 78)
    print(titulo)
    print("=" * 78)


def guardar_json(obj: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)


def guardar_csv(df, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def main() -> None:
    t_inicio = time.time()
    set_global_seed(config.RANDOM_SEED)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Datos
    # -----------------------------------------------------------------
    seccion("1. Carga, auditoría y limpieza de datos")
    train, test, audit_train, audit_test = data.load_train_test()
    print(audit_train.resumen_texto())
    print()
    print(audit_test.resumen_texto())

    episodios_train = data.episode_summary(train)
    episodios_test = data.episode_summary(test)
    guardar_csv(episodios_train, config.TABLES_DIR / "episodios_train.csv")
    guardar_csv(episodios_test, config.TABLES_DIR / "episodios_test.csv")

    guardar_json(
        {"train": audit_train.to_dict(), "test": audit_test.to_dict()},
        config.TABLES_DIR / "auditoria_datos.json",
    )

    test_context = data.build_warmup_context(train, test)
    n_tradable_segmento_a = int(test_context.loc[test_context["episode_id"] == test_context["episode_id"].iloc[0]]
                                 .query("is_test_row")["is_tradable"].sum())
    segundo_episodio_test = test_context["episode_id"].unique()[-1]
    sub_b = test_context[test_context["episode_id"] == segundo_episodio_test]
    n_tradable_segmento_b = int(sub_b["is_tradable"].sum())
    print(f"\nBarras operables en el segmento de test del 2023-12-31 (con cola de train): {n_tradable_segmento_a}")
    print(f"Barras operables en el segmento de test de mayo-junio 2024 (calentamiento propio): {n_tradable_segmento_b}"
          f" de {len(sub_b)} totales")

    # -----------------------------------------------------------------
    # 2. Walk-forward sobre train
    # -----------------------------------------------------------------
    seccion("2. Walk-forward (1 mes train / 1 semana test / paso semanal) sobre train")
    wf = optimize.run_walkforward(train, seed=config.RANDOM_SEED, verbose=True)
    print(f"\nVentanas procesadas: {wf.windows_table['window'].nunique()}")
    print(f"Configuraciones evaluadas en total: {wf.n_configs_evaluated}")
    print(f"Tiempo de optimización: {wf.elapsed_seconds:.1f} s")

    guardar_csv(wf.windows_table, config.TABLES_DIR / "walkforward_ventanas.csv")
    guardar_csv(wf.silhouette_by_window, config.TABLES_DIR / "silhouette_por_ventana.csv")

    validos = wf.windows_table.dropna(subset=["wfe"])
    wfe_media = validos["wfe"].mean() if len(validos) else float("nan")
    print(f"\nWFE media (ventanas con Calmar IS > 0.05): {wfe_media:.3f} sobre {len(validos)} observaciones"
          f" de {len(wf.windows_table)} totales")

    # -----------------------------------------------------------------
    # 3. Congelamiento ANTES de tocar el test
    # -----------------------------------------------------------------
    seccion("3. Congelamiento de parámetros y modelo de régimen")
    congelado = optimize.freeze_final_config(wf)
    guardar_json(congelado, config.TABLES_DIR / "parametros_congelados.json")
    print("Parámetros y modelo de régimen escritos en docs/tablas/parametros_congelados.json")
    for nombre, p in congelado["params_by_regime"].items():
        print(f"  {nombre}: {p}")

    # -----------------------------------------------------------------
    # 4. Evaluación en test (una sola vez)
    # -----------------------------------------------------------------
    seccion("4. Evaluación final en test (una sola vez, sin reoptimizar)")
    test_eval = optimize.evaluate_on_test(train, test_context, wf.regime_model_last, wf.params_by_regime_last)
    equity_test = test_eval["combinado"].equity
    trades_test = test_eval["combinado"].trades_frame()
    print(f"Capital final en test: ${equity_test.iloc[-1]:,.2f} (inicial ${config.INITIAL_CAPITAL:,.2f})")
    print(f"Operaciones en test: {len(trades_test)}")

    # -----------------------------------------------------------------
    # 5. Métricas
    # -----------------------------------------------------------------
    seccion("5. Métricas de desempeño")
    train_oos_equity = wf.oos_equity
    train_oos_trades = wf.oos_trades
    precios_train_indexados = train.drop_duplicates("Datetime").set_index("Datetime")["Close"]
    df_bench_train = pd.DataFrame({"Close": precios_train_indexados.reindex(train_oos_equity.index).ffill()})
    benchmark_train = metrics.benchmark_equity(df_bench_train)
    benchmark_test = metrics.benchmark_equity(test_context[test_context["is_test_row"]])

    tabla_train = metrics.summary_table(train_oos_equity, train_oos_trades, "train_oos_walkforward", benchmark_train)
    tabla_test = metrics.summary_table(equity_test, trades_test, "test_completo", benchmark_test)

    segmentos_tabla = [tabla_train, tabla_test]
    for ep_id, seg in test_eval["por_segmento"].items():
        seg_trades = seg["result"].trades_frame()
        nombre = "test_segmento_dic2023" if ep_id == list(test_eval["por_segmento"].keys())[0] else "test_segmento_may2024"
        segmentos_tabla.append(metrics.summary_table(seg["result"].equity, seg_trades, nombre))

    tabla_metricas = pd.concat(segmentos_tabla, ignore_index=True)
    guardar_csv(tabla_metricas, config.TABLES_DIR / "metricas_resumen.csv")
    print(tabla_metricas.to_string(index=False))

    for calmar_val, nombre in [(tabla_train["sharpe"].iloc[0], "train OOS"), (tabla_test["sharpe"].iloc[0], "test")]:
        if pd.notna(calmar_val) and calmar_val > 3:
            print(f"ALERTA: Sharpe > 3 en {nombre} ({calmar_val:.2f}); revisar posible look-ahead.")

    tablas_retornos_train = metrics.monthly_quarterly_annual_tables(train_oos_equity)
    tablas_retornos_test = metrics.monthly_quarterly_annual_tables(equity_test)
    for freq, serie in tablas_retornos_train.items():
        guardar_csv(serie, config.TABLES_DIR / f"retornos_{freq}_train_oos.csv")
    for freq, serie in tablas_retornos_test.items():
        guardar_csv(serie, config.TABLES_DIR / f"retornos_{freq}_test.csv")

    # -----------------------------------------------------------------
    # 6. Validación de régimen
    # -----------------------------------------------------------------
    seccion("6. Validación de régimen")
    dur_train = regimes.regime_duration_stats(wf.regime_codes_oos)
    dur_test = regimes.regime_duration_stats(test_eval["regimen_total"])
    share_train = regimes.time_share_by_regime(wf.regime_codes_oos)
    share_test = regimes.time_share_by_regime(test_eval["regimen_total"])
    print("Duración media por régimen (train, OOS walk-forward):")
    print(dur_train)
    print("\nDuración media por régimen (test):")
    print(dur_test)
    print("\nProporción de tiempo por régimen, train vs test:")
    print(pd.DataFrame({"train": share_train, "test": share_test}))

    guardar_csv(dur_train, config.TABLES_DIR / "regimen_duracion_train.csv")
    guardar_csv(dur_test, config.TABLES_DIR / "regimen_duracion_test.csv")
    guardar_csv(pd.DataFrame({"train": share_train, "test": share_test}), config.TABLES_DIR / "regimen_proporcion_tiempo.csv")

    silhouette_media = wf.silhouette_by_window["silhouette"].mean()
    print(f"\nSilhouette promedio a lo largo del walk-forward: {silhouette_media:.3f}")
    print(f"Silhouette de la última ventana (congelada): {congelado['regime_model']['silhouette_ultima_ventana']:.3f}")

    diferenciacion = []
    for code, nombre in enumerate(config.REGIME_NAMES):
        sub = train_oos_trades[train_oos_trades["regime_entry"] == code]
        diferenciacion.append({
            "regimen": nombre, "n_operaciones": len(sub),
            "pnl_promedio": sub["pnl"].mean() if len(sub) else np.nan,
            "win_rate": metrics.win_rate(sub) if len(sub) else np.nan,
        })
    tabla_diferenciacion = pd.DataFrame(diferenciacion)
    guardar_csv(tabla_diferenciacion, config.TABLES_DIR / "regimen_diferenciacion_desempeno.csv")
    print("\nDiferenciación de desempeño por régimen (operaciones OOS de train):")
    print(tabla_diferenciacion)

    pnl_tendencia = train_oos_trades.loc[train_oos_trades["regime_entry"] == 0, "pnl"]
    pnl_reversion = train_oos_trades.loc[train_oos_trades["regime_entry"] == 1, "pnl"]
    prueba_bootstrap = metrics.bootstrap_difference_test(pnl_tendencia, pnl_reversion)
    guardar_json(prueba_bootstrap, config.TABLES_DIR / "regimen_bootstrap_tendencia_vs_reversion.json")
    print("\nPrueba de bootstrap (PnL tendencia vs reversión):", prueba_bootstrap)

    # -----------------------------------------------------------------
    # 7. Robustez
    # -----------------------------------------------------------------
    seccion("7. Robustez")
    sens_df = optimize.sensitivity_analysis(wf.window_cache, wf.params_by_regime_last)
    guardar_csv(sens_df, config.TABLES_DIR / "sensibilidad_parametros.csv")

    niveles_costo = np.concatenate([
        np.linspace(0.0, config.COMMISSION * 1.0, 3),
        np.array([config.COMMISSION * m for m in config.COST_MULTIPLIERS]),
        np.linspace(config.COMMISSION * 4, config.COMMISSION * 20, 10),
    ])
    niveles_costo = np.unique(np.round(niveles_costo, 6))
    cost_df = optimize.cost_sensitivity_sweep(wf.window_cache, wf.params_by_regime_last, niveles_costo)
    guardar_csv(cost_df, config.TABLES_DIR / "sensibilidad_costos.csv")

    rentables = cost_df[cost_df["retorno_total"] > 0]
    costo_equilibrio = rentables["comision"].max() if len(rentables) else 0.0
    margen_seguridad = (costo_equilibrio - config.COMMISSION) / config.COMMISSION if costo_equilibrio else float("nan")
    print(f"Costo de equilibrio aproximado: {costo_equilibrio:.4%} | margen de seguridad vs 0.125%: {margen_seguridad:.1%}")

    abl_df = optimize.single_indicator_comparison(wf.window_cache, wf.params_by_regime_last)
    guardar_csv(abl_df, config.TABLES_DIR / "comparacion_confirmacion_vs_individual.csv")
    print(abl_df)

    friccion = metrics.estimate_execution_friction(trades_test, test_context[test_context["is_test_row"]]["Close"],
                                                    config.ASSUMED_DAILY_DOLLAR_VOLUME)
    guardar_json(friccion, config.TABLES_DIR / "friccion_ejecucion_test.json")
    print("\nEstimación de fricción de ejecución (test):", friccion)

    # -----------------------------------------------------------------
    # 8. Figuras
    # -----------------------------------------------------------------
    seccion("8. Figuras")
    plots.plot_equity_vs_benchmark(train_oos_equity, benchmark_train,
                                    "Valor del portafolio - Train (OOS walk-forward) vs Buy & Hold",
                                    config.FIGURES_DIR / "fig01a_equity_train.png")
    plots.plot_equity_vs_benchmark(equity_test, benchmark_test,
                                    "Valor del portafolio - Test vs Buy & Hold",
                                    config.FIGURES_DIR / "fig01b_equity_test.png")
    plots.plot_drawdown(train_oos_equity, "Drawdown - Train (OOS walk-forward)",
                         config.FIGURES_DIR / "fig02a_drawdown_train.png")
    plots.plot_drawdown(equity_test, "Drawdown - Test", config.FIGURES_DIR / "fig02b_drawdown_test.png")

    plots.plot_returns_heatmap(tablas_retornos_train["mensual"], "Retornos mensuales - Train OOS",
                                config.FIGURES_DIR / "fig03a_retornos_mensuales_train.png")
    plots.plot_returns_heatmap(tablas_retornos_test["mensual"], "Retornos mensuales - Test",
                                config.FIGURES_DIR / "fig03b_retornos_mensuales_test.png", date_fmt="%Y-%m-%d")

    plots.plot_sensitivity(sens_df, "Sensibilidad de parámetros óptimos (±20%)",
                            config.FIGURES_DIR / "fig04_sensibilidad_parametros.png")
    plots.plot_cost_curve(cost_df, "Retorno neto vs nivel de costo de transacción",
                           config.FIGURES_DIR / "fig05_retorno_vs_costo.png")

    precio_train_oos = train[train["Datetime"].isin(wf.regime_codes_oos.index)]
    plots.plot_regime_timeline(precio_train_oos.drop_duplicates("Datetime"), wf.regime_codes_oos,
                                "Línea de tiempo de régimen sobre el precio (train, OOS)",
                                config.FIGURES_DIR / "fig06a_regimen_timeline_train.png")

    ultima_ventana = wf.window_cache[-1]
    features_ultima = regimes.compute_regime_features(ultima_ventana["context_df"])
    plots.plot_regime_variable_distributions(features_ultima, ultima_ventana["held"],
                                              "Distribución de variables de régimen (última ventana de train)",
                                              config.FIGURES_DIR / "fig06b_distribuciones_regimen.png")
    plots.plot_equity_with_regimes(train_oos_equity, wf.regime_codes_oos,
                                    "Valor del portafolio con regímenes superpuestos (train, OOS)",
                                    config.FIGURES_DIR / "fig06c_equity_regimenes.png")

    for r, nombre in enumerate(config.REGIME_NAMES):
        study = wf.last_window_studies.get(r)
        if study is None or len(study.trials) == 0:
            continue
        plots.plot_optimization_history(study, f"Historia de optimización - régimen {nombre} (última ventana)",
                                         config.FIGURES_DIR / f"fig07_historia_{nombre}.png")
        plots.plot_param_importance(study, f"Importancia de hiperparámetros - régimen {nombre}",
                                     config.FIGURES_DIR / f"fig08_importancia_{nombre}.png")
        plots.plot_slice(study, ["sl_atr_mult", "tp_atr_mult", "position_fraction"],
                          f"Slice plots - régimen {nombre}", config.FIGURES_DIR / f"fig09_slice_{nombre}.png")
        plots.plot_2d_surface(study, "sl_atr_mult", "tp_atr_mult",
                               f"Superficie de Calmar (SL x TP, resto fijo) - régimen {nombre}",
                               config.FIGURES_DIR / f"fig10_superficie_{nombre}.png")

    elapsed_total = time.time() - t_inicio
    seccion("Resumen final")
    print(f"Tiempo total de ejecución: {elapsed_total/60:.1f} minutos")
    print(f"Retorno total train (OOS walk-forward): {tabla_train['retorno_total'].iloc[0]:.2%}")
    print(f"Retorno total test: {tabla_test['retorno_total'].iloc[0]:.2%}")
    print(f"Calmar train (OOS): {tabla_train['calmar'].iloc[0]:.3f} | Calmar test: {tabla_test['calmar'].iloc[0]:.3f}")
    print("Listo. Ver docs/figuras y docs/tablas para el detalle completo.")


if __name__ == "__main__":
    main()
