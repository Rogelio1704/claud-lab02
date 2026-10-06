"""Punto de entrada único del proyecto: ``python main.py``.

Ejecuta, en orden, todo el pipeline CORREGIDO del Laboratorio 02 (Nivel B).
Una primera corrida completa reveló defectos de lógica y de análisis (ver
``docs/reporte.pdf``, sección "Corrección post-revisión"): la función
objetivo premiaba la ruina, el espacio de búsqueda de take-profit no
contenía configuraciones capaces de cubrir el costo de transacción, el
cierre por cambio de régimen se adelantaba una barra, la diferenciación por
régimen se medía en dólares en vez de en retorno porcentual, no había una
corrida de referencia sin régimen, y el benchmark de comprar y mantener
cruzaba gratis el hueco de 122 días que la estrategia tiene prohibido
cruzar. Todas las correcciones se motivaron con diagnósticos de TRAIN, no
con el resultado de test: el test se evalúa una sola vez, al final, con los
parámetros ya congelados.

1. Carga, audita y limpia los datos de train y test.
2. Calcula el diagnóstico de costo de ida y vuelta contra ATR (corrección 1.2).
3. Corre el walk-forward CON régimen (función objetivo corregida).
4. Corre el walk-forward SIN régimen (referencia, pregunta de análisis 5).
5. Congela los parámetros finales y el modelo de régimen ANTES de tocar el test.
6. Evalúa, una sola vez, el test completo y cada uno de sus dos segmentos.
7. Calcula métricas (con benchmark encadenado por episodios) y validación de régimen.
8. Corre el análisis de robustez (sensibilidad ±20%, costos, comparación 2 de 3,
   fricción de ejecución en rango).
9. Genera las figuras y tablas en ``docs/figuras`` y ``docs/tablas``.

No recibe argumentos. El tiempo aproximado de ejecución y la semilla usada
se documentan en el README.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import pandas as pd

from src import config, data, metrics, optimize, plots, regimes, signals


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


def resumen_wfe(windows_table: pd.DataFrame) -> dict:
    """Walk-Forward Efficiency con el detalle pedido en la corrección 2.2:
    cuántas observaciones la sostienen, qué proporción de las combinaciones
    ventana-régimen tuvo Calmar OOS positivo, y las medianas de Calmar IS y
    OOS (más robustas que la media frente a los valores extremos que puede
    tomar un Calmar calculado sobre pocas operaciones)."""
    validos = windows_table.dropna(subset=["wfe"])
    con_oos = windows_table.dropna(subset=["calmar_oos"])
    return {
        "wfe_media": float(validos["wfe"].mean()) if len(validos) else float("nan"),
        "wfe_mediana": float(validos["wfe"].median()) if len(validos) else float("nan"),
        "n_observaciones_wfe": int(len(validos)),
        "n_combinaciones_totales": int(len(windows_table)),
        "pct_calmar_oos_positivo": float((con_oos["calmar_oos"] > 0).mean()) if len(con_oos) else float("nan"),
        "n_combinaciones_con_oos": int(len(con_oos)),
        "calmar_is_mediana": float(windows_table["calmar_is"].median()),
        "calmar_oos_mediana": float(con_oos["calmar_oos"].median()) if len(con_oos) else float("nan"),
    }


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
    # 2. Diagnóstico: costo de ida y vuelta contra ATR (corrección 1.2)
    # -----------------------------------------------------------------
    seccion("2. Diagnóstico: costo de ida y vuelta contra ATR (datos reales de train)")
    diagnostico_atr = metrics.atr_cost_diagnostic(train, atr_window=16, commission=config.COMMISSION)
    print(f"Costo de ida y vuelta: {diagnostico_atr['costo_ida_vuelta']:.3%}")
    print(f"ATR(16)/precio en train: mediana={diagnostico_atr['atr_frac_precio_mediana']:.4%} "
          f"(p25={diagnostico_atr['atr_frac_precio_p25']:.4%}, p75={diagnostico_atr['atr_frac_precio_p75']:.4%})")
    print(f"Múltiplo de ATR para apenas cubrir el costo (mediana): {diagnostico_atr['tp_mult_equilibrio_mediana']:.2f}")
    print(f"Rango corregido de búsqueda: sl_atr_mult = {signals.PARAM_BOUNDS['sl_atr_mult']}, "
          f"tp_atr_mult = {signals.PARAM_BOUNDS['tp_atr_mult']}")
    guardar_json(diagnostico_atr, config.TABLES_DIR / "diagnostico_atr_costo.json")

    # -----------------------------------------------------------------
    # 3. Walk-forward CON régimen
    # -----------------------------------------------------------------
    seccion("3. Walk-forward CON régimen (1 mes train / 1 semana test / paso semanal)")
    windows_generadas = optimize.generate_walkforward_windows(train)
    wf = optimize.run_walkforward(train, seed=config.RANDOM_SEED, verbose=True)
    n_generadas = len(windows_generadas)
    n_procesadas = wf.windows_table["window"].nunique()
    print(f"\nVentanas generadas: {n_generadas} | procesadas: {n_procesadas} | "
          f"omitidas: {n_generadas - n_procesadas} (no alcanzaron el mínimo de filas con variables de "
          f"régimen completas —N_REGIMES*10— para ajustar K-means en esa ventana)")
    print(f"Configuraciones evaluadas: {wf.n_configs_evaluated} | tiempo de optimización: {wf.elapsed_seconds/60:.1f} min")

    guardar_csv(wf.windows_table, config.TABLES_DIR / "walkforward_ventanas.csv")
    guardar_csv(wf.silhouette_by_window, config.TABLES_DIR / "silhouette_por_ventana.csv")

    wfe_info = resumen_wfe(wf.windows_table)
    guardar_json(wfe_info, config.TABLES_DIR / "wfe_resumen.json")
    print(f"\nWFE media={wfe_info['wfe_media']:.3f} | mediana={wfe_info['wfe_mediana']:.3f} "
          f"sobre {wfe_info['n_observaciones_wfe']} de {wfe_info['n_combinaciones_totales']} combinaciones.")
    print(f"Combinaciones ventana-régimen con Calmar OOS > 0: {wfe_info['pct_calmar_oos_positivo']:.1%} "
          f"de {wfe_info['n_combinaciones_con_oos']} con OOS evaluado.")
    print(f"Calmar IS mediana: {wfe_info['calmar_is_mediana']:.3f} | Calmar OOS mediana: {wfe_info['calmar_oos_mediana']:.3f}")

    # -----------------------------------------------------------------
    # 4. Walk-forward SIN régimen (referencia; pregunta de análisis 5)
    # -----------------------------------------------------------------
    seccion("4. Walk-forward SIN régimen (referencia: un solo conjunto de parámetros por ventana)")
    wf_baseline = optimize.run_walkforward_baseline(train, seed=config.RANDOM_SEED, verbose=True)
    n_procesadas_baseline = wf_baseline.windows_table["window"].nunique()
    print(f"\nVentanas generadas: {n_generadas} | procesadas: {n_procesadas_baseline} | "
          f"omitidas: {n_generadas - n_procesadas_baseline} (no necesita ajustar régimen, así que solo se "
          "omiten ventanas sin ninguna barra operable, lo que no ocurrió en este caso)")
    print(f"Configuraciones evaluadas: {wf_baseline.n_configs_evaluated} | "
          f"tiempo de optimización: {wf_baseline.elapsed_seconds/60:.1f} min")
    guardar_csv(wf_baseline.windows_table, config.TABLES_DIR / "walkforward_baseline_ventanas.csv")

    wfe_baseline_info = resumen_wfe(wf_baseline.windows_table)
    guardar_json(wfe_baseline_info, config.TABLES_DIR / "wfe_baseline_resumen.json")

    n_configs_total_pipeline = wf.n_configs_evaluated + wf_baseline.n_configs_evaluated
    tiempo_opt_total = wf.elapsed_seconds + wf_baseline.elapsed_seconds
    print(f"\nConfiguraciones evaluadas en TODO el pipeline (con régimen + referencia): "
          f"{n_configs_total_pipeline} en {tiempo_opt_total/60:.1f} min")

    tabla_baseline_vs_regimen = pd.DataFrame([
        {
            "version": "con_regimen", "retorno_total": wf.oos_equity.iloc[-1] / wf.oos_equity.iloc[0] - 1.0,
            "calmar": metrics.calmar_ratio(wf.oos_equity), "n_operaciones": len(wf.oos_trades),
            "wfe_media": wfe_info["wfe_media"], "pct_calmar_oos_positivo": wfe_info["pct_calmar_oos_positivo"],
        },
        {
            "version": "sin_regimen", "retorno_total": wf_baseline.oos_equity.iloc[-1] / wf_baseline.oos_equity.iloc[0] - 1.0,
            "calmar": metrics.calmar_ratio(wf_baseline.oos_equity), "n_operaciones": len(wf_baseline.oos_trades),
            "wfe_media": wfe_baseline_info["wfe_media"], "pct_calmar_oos_positivo": wfe_baseline_info["pct_calmar_oos_positivo"],
        },
    ])
    guardar_csv(tabla_baseline_vs_regimen, config.TABLES_DIR / "comparacion_con_vs_sin_regimen.csv")
    print("\nComparación con régimen vs sin régimen (trayectoria OOS de train; no se evalúa en test "
          "para no multiplicar el uso del conjunto de prueba):")
    print(tabla_baseline_vs_regimen.to_string(index=False))

    # -----------------------------------------------------------------
    # 5. Congelamiento ANTES de tocar el test
    # -----------------------------------------------------------------
    seccion("5. Congelamiento de parámetros y modelo de régimen")
    congelado = optimize.freeze_final_config(wf)
    guardar_json(congelado, config.TABLES_DIR / "parametros_congelados.json")
    print("Parámetros y modelo de régimen escritos en docs/tablas/parametros_congelados.json")
    for nombre, p in congelado["params_by_regime"].items():
        print(f"  {nombre}: {p}")

    # -----------------------------------------------------------------
    # 6. Evaluación en test (una sola vez)
    # -----------------------------------------------------------------
    seccion("6. Evaluación final en test (una sola vez, sin reoptimizar)")
    test_eval = optimize.evaluate_on_test(train, test_context, wf.regime_model_last, wf.params_by_regime_last)
    equity_test = test_eval["combinado"].equity
    trades_test = test_eval["combinado"].trades_frame()
    print(f"Capital final en test: ${equity_test.iloc[-1]:,.2f} (inicial ${config.INITIAL_CAPITAL:,.2f})")
    print(f"Operaciones en test: {len(trades_test)}")

    # -----------------------------------------------------------------
    # 7. Métricas (benchmark corregido: encadenado por episodios, 1.6)
    # -----------------------------------------------------------------
    seccion("7. Métricas de desempeño")
    train_oos_equity = wf.oos_equity
    train_oos_trades = wf.oos_trades

    precios_oos_train = pd.concat(
        [item["context_df"].loc[item["oos_mask"], ["Datetime", "Close", "episode_id"]] for item in wf.window_cache],
        ignore_index=True,
    )
    benchmark_train = metrics.benchmark_equity_chained(precios_oos_train)
    benchmark_test = metrics.benchmark_equity_chained(
        test_context.loc[test_context["is_test_row"], ["Datetime", "Close", "episode_id"]]
    )

    tabla_train = metrics.summary_table(train_oos_equity, train_oos_trades, "train_oos_walkforward", benchmark_train)
    tabla_test = metrics.summary_table(equity_test, trades_test, "test_completo", benchmark_test)

    segmentos_tabla = [tabla_train, tabla_test]
    claves_segmento = list(test_eval["por_segmento"].keys())
    for ep_id, seg in test_eval["por_segmento"].items():
        seg_trades = seg["result"].trades_frame()
        nombre = "test_segmento_dic2023" if ep_id == claves_segmento[0] else "test_segmento_may2024"
        segmentos_tabla.append(metrics.summary_table(seg["result"].equity, seg_trades, nombre))

    tabla_metricas = pd.concat(segmentos_tabla, ignore_index=True)
    guardar_csv(tabla_metricas, config.TABLES_DIR / "metricas_resumen.csv")
    print(tabla_metricas.to_string(index=False))

    sharpes_a_revisar = [("train OOS", tabla_train["sharpe"].iloc[0]), ("test", tabla_test["sharpe"].iloc[0])]
    for nombre, valor in sharpes_a_revisar:
        if pd.notna(valor) and valor > 3:
            print(f"ALERTA: Sharpe > 3 en {nombre} ({valor:.2f}); revisar posible look-ahead antes de reportar.")
        else:
            print(f"Sharpe en {nombre}: {valor:.2f} (sin alerta; no supera 3)")

    tablas_retornos_train = metrics.monthly_quarterly_annual_tables(train_oos_equity)
    tablas_retornos_test = metrics.monthly_quarterly_annual_tables(equity_test)
    for freq, serie in tablas_retornos_train.items():
        guardar_csv(serie, config.TABLES_DIR / f"retornos_{freq}_train_oos.csv")
    for freq, serie in tablas_retornos_test.items():
        guardar_csv(serie, config.TABLES_DIR / f"retornos_{freq}_test.csv")

    # -----------------------------------------------------------------
    # 8. Validación de régimen (diferenciación en % de retorno, 1.4)
    # -----------------------------------------------------------------
    seccion("8. Validación de régimen")
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
    silhouette_ultima = congelado["regime_model"]["silhouette_ultima_ventana"]
    cumple_silhouette = silhouette_media > 0.4
    print(f"\nSilhouette promedio: {silhouette_media:.3f} | última ventana (congelada): {silhouette_ultima:.3f} "
          f"-> objetivo (>0.4): {'SE CUMPLE' if cumple_silhouette else 'NO se cumple'} en promedio")

    duracion_min_train = dur_train["duracion_media_horas"].min()
    cumple_persistencia = duracion_min_train > 12
    print(f"Persistencia: duración media mínima entre regímenes en train = {duracion_min_train:.1f} h "
          f"-> objetivo (>12h): {'SE CUMPLE' if cumple_persistencia else 'NO se cumple'}")
    validacion_regimen = {
        "silhouette_promedio": float(silhouette_media), "silhouette_ultima_ventana": float(silhouette_ultima),
        "objetivo_silhouette_cumplido": bool(cumple_silhouette),
        "duracion_media_minima_train_horas": float(duracion_min_train),
        "objetivo_persistencia_cumplido": bool(cumple_persistencia),
    }
    guardar_json(validacion_regimen, config.TABLES_DIR / "regimen_validacion_resumen.json")

    diferenciacion = []
    for code, nombre in enumerate(config.REGIME_NAMES):
        sub = train_oos_trades[train_oos_trades["regime_entry"] == code]
        retornos_pct = metrics.trade_returns_pct(sub)
        diferenciacion.append({
            "regimen": nombre, "n_operaciones": len(sub),
            "retorno_pct_promedio": float(retornos_pct.mean()) if len(sub) else np.nan,
            "win_rate": metrics.win_rate(sub) if len(sub) else np.nan,
        })
    tabla_diferenciacion = pd.DataFrame(diferenciacion)
    guardar_csv(tabla_diferenciacion, config.TABLES_DIR / "regimen_diferenciacion_desempeno.csv")
    print("\nDiferenciación de desempeño por régimen, en retorno % por operación (operaciones OOS de train):")
    print(tabla_diferenciacion)

    retpct_tendencia = metrics.trade_returns_pct(train_oos_trades[train_oos_trades["regime_entry"] == 0])
    retpct_reversion = metrics.trade_returns_pct(train_oos_trades[train_oos_trades["regime_entry"] == 1])
    prueba_bootstrap = metrics.bootstrap_difference_test(retpct_tendencia, retpct_reversion)
    guardar_json(prueba_bootstrap, config.TABLES_DIR / "regimen_bootstrap_tendencia_vs_reversion.json")
    print("\nPrueba de bootstrap sobre retorno % por operación (tendencia vs reversión):", prueba_bootstrap)

    # -----------------------------------------------------------------
    # 9. Robustez
    # -----------------------------------------------------------------
    seccion("9. Robustez")
    sens_df = optimize.sensitivity_analysis(wf.window_cache, wf.params_by_regime_last)
    guardar_csv(sens_df, config.TABLES_DIR / "sensibilidad_parametros.csv")
    print(f"Sensibilidad ±20%: {len(sens_df)} filas (Calmar, retorno total y N operaciones por variante). "
          f"Dispersión del Calmar entre variantes: {sens_df['calmar'].std():.3f}")

    niveles_costo = np.concatenate([
        np.linspace(0.0, config.COMMISSION * 1.0, 9),  # resolución fina entre 0% y 0.125%, donde pasó a estar el equilibrio
        np.array([config.COMMISSION * m for m in config.COST_MULTIPLIERS]),
        np.linspace(config.COMMISSION * 4, config.COMMISSION * 20, 10),
    ])
    niveles_costo = np.unique(np.round(niveles_costo, 6))
    cost_df = optimize.cost_sensitivity_sweep(wf.window_cache, wf.params_by_regime_last, niveles_costo)
    guardar_csv(cost_df, config.TABLES_DIR / "sensibilidad_costos.csv")

    rentables = cost_df[cost_df["retorno_total"] > 0]
    if len(rentables) > 0:
        costo_equilibrio = rentables["comision"].max()
        margen_seguridad = (costo_equilibrio - config.COMMISSION) / config.COMMISSION
    else:
        costo_equilibrio = float("nan")
        margen_seguridad = float("nan")
    print(f"Costo de equilibrio aproximado: {costo_equilibrio:.4%} | margen de seguridad vs 0.125%: {margen_seguridad:.1%}")

    abl_df = optimize.single_indicator_comparison(wf.window_cache, wf.params_by_regime_last)
    guardar_csv(abl_df, config.TABLES_DIR / "comparacion_confirmacion_vs_individual.csv")
    if abl_df["calmar"].nunique() <= 1 and abl_df["retorno_total"].nunique() <= 1:
        print("Comparación 2 de 3 vs un solo indicador: la prueba NO discrimina (mismo Calmar/retorno en todas "
              "las variantes).")
    print(abl_df)

    tabla_friccion, friccion = metrics.estimate_execution_friction_grid(trades_test, test_context.loc[test_context["is_test_row"], "Close"])
    guardar_csv(tabla_friccion, config.TABLES_DIR / "friccion_ejecucion_rejilla.csv")
    guardar_json(friccion, config.TABLES_DIR / "friccion_ejecucion_test.json")
    print(f"\nFricción de ejecución en test (rango Y={config.MARKET_IMPACT_Y_RANGE}, "
          f"volumen={config.ASSUMED_DAILY_DOLLAR_VOLUME_RANGE}): "
          f"costo total entre ${friccion['costo_friccion_min']:,.0f} y ${friccion['costo_friccion_max']:,.0f}; "
          f"PnL recalculado entre ${friccion['pnl_con_friccion_min']:,.0f} y ${friccion['pnl_con_friccion_max']:,.0f} "
          f"(PnL original solo con comisión: ${friccion['pnl_original']:,.0f}).")

    # -----------------------------------------------------------------
    # 10. Figuras
    # -----------------------------------------------------------------
    seccion("10. Figuras")
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
                               f"Superficie del objetivo (SL x TP, resto fijo) - régimen {nombre}",
                               config.FIGURES_DIR / f"fig10_superficie_{nombre}.png")

    elapsed_total = time.time() - t_inicio
    seccion("Resumen final")
    print(f"Tiempo total de ejecución: {elapsed_total/60:.1f} minutos")
    print(f"Retorno total train (OOS walk-forward): {tabla_train['retorno_total'].iloc[0]:.2%}")
    print(f"Retorno total test: {tabla_test['retorno_total'].iloc[0]:.2%}")
    print(f"Calmar train (OOS): {tabla_train['calmar'].iloc[0]:.3f} | Calmar test: {tabla_test['calmar'].iloc[0]:.3f}")
    print(f"Win rate train: {tabla_train['win_rate'].iloc[0]:.2%} | Win rate test: {tabla_test['win_rate'].iloc[0]:.2%}")
    print("Listo. Ver docs/figuras y docs/tablas para el detalle completo.")


if __name__ == "__main__":
    main()
