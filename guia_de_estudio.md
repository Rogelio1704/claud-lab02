# Guía de estudio — Laboratorio 02 (Nivel B)

Esta guía explica, en lenguaje llano, qué hace cada módulo, por qué se tomó cada decisión de
diseño, dónde ocurre en el código, y qué preguntas es probable que haga el profesor con sus
respuestas. Está pensada para que cualquiera de los dos integrantes pueda sostener cualquier
parte del proyecto sin haber escrito esa línea en particular.

## 1. Panorama general

El pipeline completo corre con `python main.py` y hace, en orden:

1. Carga y audita los dos CSV, los limpia de forma causal (`src/data.py`).
2. Corre un walk-forward sobre `train` (1 mes de entrenamiento, 1 semana de prueba, paso
   semanal) que, en cada ventana, ajusta un modelo de régimen (`src/regimes.py`) y optimiza un
   conjunto de hiperparámetros por régimen (`src/optimize.py`) usando el motor de backtesting
   (`src/backtest.py`) y las métricas (`src/metrics.py`).
3. Congela los parámetros y el modelo de régimen de la última ventana, y evalúa `test` una
   sola vez.
4. Corre el análisis de robustez (sensibilidad, costos, comparación de indicadores, fricción
   de ejecución).
5. Genera todas las figuras (`src/plots.py`) y tablas.

La cadena conceptual, de principio a fin, es: **indicador → señal → confirmación → sizing →
posición → salida (SL/TP/señal contraria/cambio de régimen/hueco/fin de periodo) → métricas →
optimización → walk-forward → congelamiento → test**. Esa es exactamente la cadena que viene
en los apuntes de clase.

## 2. `src/config.py` — constantes y supuestos

**Qué hace:** centraliza TODO lo que podría cambiar: la semilla (42), la comisión (0.125%), el
capital inicial, el umbral de hueco (30 minutos), la ventana de régimen (2016 barras = 1
semana), la cadencia de actualización de régimen (2 horas), el mínimo de operaciones (8), el
número de pruebas de Optuna (40 random + 80 TPE = 120).

**Por qué:** la rúbrica exige que la semilla esté fijada "en un solo lugar". Si cada módulo
tuviera su propia constante, bastaría con que alguien olvidara actualizar una copia para romper
la reproducibilidad.

**Decisiones que probablemente te pregunten:**

- *¿Por qué 30 minutos como umbral de hueco?* Se verificó empíricamente (ver auditoría) que no
  existe ninguna diferencia entre marcas de tiempo CRUDAS entre 5 y 30 minutos en ninguno de
  los dos archivos; con 30 minutos se capturan exactamente los tres huecos reales conocidos
  (~65 min y ~48 h en train, ~122 días en test) y nada más.
- *¿Por qué las rachas de NaN (hasta 22.5 h) no se tratan como huecos?* Porque la fila SÍ
  existe en la rejilla (solo el precio está vacío): es una falla de reporte del proveedor, no
  evidencia de que el mercado se detuvo. Tratar cada una de las 126 rachas >1h como un hueco
  real habría destruido casi todo el histórico disponible para calentamiento (126 × 2016 barras
  es más grande que todo el dataset). Además, la convención de "ejecutar al precio de apertura
  si la barra abre más allá del nivel de SL/TP" ya protege contra un llenado irreal si el precio
  se movió mucho durante la racha.
- *¿Por qué N_MIN_TRADES=8 y no 30 (el valor que usó el profesor antes)?* Porque aquí las
  ventanas son de un mes y, además, por régimen: un régimen minoritario (crisis, típicamente
  ~10-15% del tiempo) puede quedarse con unos cientos de barras en una ventana de un mes.
  Exigir 30 operaciones eliminaría casi todas las combinaciones ventana-régimen. Es
  configurable y se declara explícitamente.

## 3. `src/data.py` — carga, auditoría y limpieza causal

**Qué hace:** `load_raw_csv` lee el CSV tal cual; `audit_dataset` cuenta huecos, barras vacías,
marcas fuera de rejilla y barras planas SIN modificar nada; `clean_dataset` asigna un
`episode_id` a cada fila (según los huecos grandes de calendario) y elimina las filas con OHLC
vacío (nunca rellena); `build_warmup_context` pega la cola de train al primer episodio de test
para que ese segmento (2023-12-31) no pierda barras de calentamiento, mientras que el segundo
segmento de test (mayo-junio 2024) calienta solo con sus propias primeras 2016 barras.

**Por qué episodios:** porque el motor nunca debe simular dentro de un hueco grande. Partir los
datos en episodios y correr el backtest episodio por episodio (encadenando el capital) logra
eso sin necesidad de lógica especial dentro del motor para detectar huecos.

**Dónde:** `_assign_episodes` (línea ~140), `clean_dataset` (~155), `build_warmup_context`
(~200).

**Pregunta probable:** *¿Por qué no rellenar los precios vacíos con el último cierre?* Porque
eso inventaría información: una barra "plana" ficticia podría generar una señal de ruptura de
Bollinger falsa o un ATR artificialmente bajo. Es más conservador perder esa barra que
fabricarla.

## 4. `src/signals.py` — indicadores y confirmación

**Tres indicadores, tres familias:**

1. **Cruce de EMA (tendencia):** dirección = signo(EMA_rápida − EMA_lenta), con una banda
   muerta de `trend_band_k · ATR` para no operar sobre ruido.
2. **RSI (momento):** +1 si RSI > 50+banda, −1 si RSI < 50−banda.
3. **Ruptura de Bollinger (volatilidad):** +1 si el cierre rompe la banda superior, −1 si rompe
   la inferior.

**Regla de confirmación (fórmula):**

```
votos_largo(t) = 1{d1=+1} + 1{d2=+1} + 1{d3=+1}
votos_corto(t) = 1{d1=-1} + 1{d2=-1} + 1{d3=-1}
señal(t) = +1 si votos_largo >= 2 ; -1 si votos_corto >= 2 ; 0 en otro caso
```

**Causalidad:** todas las funciones usan `rolling`/`ewm` hacia atrás (nunca `center=True` ni
`shift(-1)`). `shift_signal_for_execution` desplaza la señal confirmada en el cierre de t para
que se "vea" en la fila t+1, que es donde el motor la ejecuta contra el `Open` de esa barra.

**Dónde:** `compute_indicator_table` (~175), `confirm_signal` (~150), `shift_signal_for_execution`
(~210).

**Pregunta probable:** *¿Por qué no usar volumen?* Porque está vacío en ~48-52% de las filas y
su escala es errática (ver auditoría): cualquier indicador basado en volumen sería, en el mejor
caso, ruido, y en el peor, una fuga de información basada en huecos del proveedor de datos.

## 5. `src/backtest.py` — el motor

**Estado explícito:** `cash`, `position_open`, `side`, `qty`, `entry_price`, `stop_loss`,
`take_profit`. Cada barra pasa por, en orden: (A) cierre forzado por cambio de régimen, (B)
cierre forzado por señal contraria, (C) nueva entrada si está plano, (D) revisión de SL/TP
contra High/Low de la misma barra (si ambos caen en la misma barra, gana el SL; si la barra
abre más allá del nivel, se ejecuta al precio de apertura), (E) marca a mercado para la curva de
equity.

**Sin apalancamiento, simetría largo/corto:** para AMBOS lados se reserva el 100% del nocional
como colateral al entrar y se libera ajustado por P&L al salir. No hay costo de préstamo para
los cortos (se declara como supuesto de simetría).

**Por qué "decisión en t−1, ejecución en t":** `build_decision_arrays` arma, por cada barra, la
señal/ATR/SL/TP/fracción del régimen vigente EN ESA barra; `shift_decision_to_execution`
desplaza todo un lugar. Esto evita que el tamaño de posición o el nivel de SL/TP de una barra se
calculen con el ATR de esa misma barra (que no se conoce hasta que cierra).

**Dónde:** `run_backtest` (el bucle principal, ~95-220), `build_decision_arrays` (~25),
`portfolio_value_check`/`total_commissions_check` (las pruebas de contabilidad, ~270-300).

**Pregunta probable:** *¿Qué pasa si SL y TP caen en la misma barra?* Gana el SL (convención
conservadora, declarada también en el PDF). *¿Por qué encadenar capital entre episodios en vez
de manejar el hueco dentro del bucle?* Porque así el bucle nunca necesita saber qué viene
después: simplemente nunca ve una barra que esté al otro lado del hueco.

## 6. `src/metrics.py` — métricas

Sharpe y Sortino se anualizan con 288 barras/día × 365 días/año (BTC opera 24/7). El Calmar usa
el retorno anualizado por **tiempo calendario transcurrido** (no por conteo de barras), para que
una ventana corta o con un hueco no distorsione la anualización. Si el MDD es 0, el Calmar se
reporta como NaN (no como infinito). También incluye `roll_spread_estimate` y
`market_impact_fraction` (sección 10 del reporte, fricción de ejecución) y
`bootstrap_difference_test` (significancia de la diferencia de desempeño entre regímenes).

**Pregunta probable:** *¿Por qué el Calmar y no el Sharpe como función objetivo?* Porque el
Sharpe penaliza igual la volatilidad al alza y a la baja; el Calmar se enfoca en el drawdown
máximo, que es lo que de verdad le importa a alguien que no quiere ver su capital caer 50%
aunque el retorno promedio sea positivo.

## 7. `src/regimes.py` — detección de régimen

**Método:** K-means (3 clusters) sobre 5 variables causales calculadas con ventana móvil de 1
semana: volatilidad realizada, ATR normalizado, fuerza de tendencia (|EMA_rápida−EMA_lenta|
escalada por volatilidad), R² de una regresión lineal del precio contra el tiempo, y
autocorrelación de orden 1 de los retornos.

**Por qué K-means y no HMM:** con HMM existe el riesgo de usar Viterbi, que reetiqueta el
pasado con información futura (el PDF lo prohíbe explícitamente). Con K-means, una vez fijos el
escalador y los centroides, la etiqueta de una barra depende solo de sus propias variables.

**Etiquetado determinista:** el cluster con mayor volatilidad promedio es "crisis"; entre los
dos restantes, el de mayor fuerza de tendencia promedio es "tendencia"; el otro es "reversión".

**Causalidad:** el escalador y los centroides se ajustan UNA VEZ por ventana de entrenamiento
(solo con datos de esa ventana); después solo se transforma/predice, nunca se reajusta. La
etiqueta se "muestrea" cada 24 barras (2 horas) y se mantiene constante entre muestras.

**Dónde:** `compute_regime_features` (~45), `fit_regime_model`/`_label_clusters` (~140-175),
`labels_with_update_cadence` (~195).

**Pregunta probable:** *¿Qué pasa si un régimen casi no aparece en una ventana?* Se hereda el
parámetro de la ventana anterior (`optimize.optimize_regime_window`, la rama
`heredado_pocas_barras`), porque no hay suficientes barras para optimizar algo con significado.

## 8. `src/optimize.py` — el corazón del proyecto

**Walk-forward:** `generate_walkforward_windows` genera ventanas de 1 mes/1 semana/paso de 1
semana, recortadas para no cruzar un hueco grande dentro de la semana de prueba.

**Por régimen, por ventana:** `optimize_regime_window` aísla las barras de un régimen dentro de
la ventana de entrenamiento (`run_regime_subset_backtest`, que calcula los indicadores con TODA
la historia causal disponible pero solo "cuenta" las barras de ese régimen) y corre un estudio de
Optuna: 40 pruebas de Random Search seguidas de 80 de TPE, **en el mismo estudio**, para que TPE
use las aleatorias como observaciones iniciales (warm-start).

**Función objetivo:** el Calmar del tramo, con `PENALTY_VALUE=-1e9` si el número de operaciones
es menor a `N_MIN_TRADES`.

**Selección de meseta (`select_plateau_params`):** en vez de tomar el argmax literal, se
discretiza el espacio de hiperparámetros en una rejilla gruesa (4 bins por parámetro) y se
busca la celda con MEJOR PROMEDIO entre las que tienen al menos 2 pruebas (una meseta real). Si
ninguna celda tiene vecinos, cae de vuelta al argmax y lo declara (`argmax_pico_aislado`).

**WFE:** `calmar_oos / calmar_is`, calculado solo cuando `calmar_is > 0.05` (para no dividir
entre un número negativo o casi cero, que no tiene interpretación económica).

**Congelamiento:** `freeze_final_config` serializa el modelo de régimen (medias y escalas del
escalador, centroides de K-means, mapeo de cluster a nombre) y los parámetros por régimen de la
ÚLTIMA ventana, a un JSON, ANTES de tocar el test.

**Robustez:** `sensitivity_analysis` perturba cada hiperparámetro ±20% (uno a la vez, por
régimen) y reevalúa sobre TODA la trayectoria OOS de train (no solo la última semana, para tener
más potencia estadística). `cost_sensitivity_sweep` barre el nivel de comisión.
`single_indicator_comparison` reemplaza la columna de señal usada (`signal`, `dir_trend`,
`dir_momentum`, `dir_vol`) para comparar 2-de-3 contra cada indicador solo.

**Dónde:** todo el archivo, pero las funciones clave son `optimize_regime_window` (~230),
`select_plateau_params` (~165), `run_walkforward` (~330), `sensitivity_analysis` (~430).

**Pregunta probable:** *¿Por qué optimizar con subconjuntos por régimen y no con una simulación
conjunta?* Por velocidad (permite 120 pruebas × 3 regímenes × 76 ventanas en minutos, no horas)
y porque así el Calmar que optimiza el parámetro de un régimen no se contamina con el
desempeño de otro régimen dentro de la misma ventana. La simulación CONJUNTA (con cambios de
régimen reales) se usa para la curva de equity final, no para elegir los parámetros.

## 9. `src/plots.py` — figuras

Todas las figuras reciben datos ya calculados (nunca vuelven a correr una simulación) y llevan
título, ejes y leyenda. Las figuras de diagnóstico de Optuna (`plot_optimization_history`,
`plot_param_importance`, `plot_slice`, `plot_2d_surface`) usan el objeto `study` de la ÚLTIMA
ventana (guardado en `wf.last_window_studies`), porque guardar el estudio completo de las 76×3
combinaciones sería demasiado pesado en memoria.

## 10. `main.py` — orquestación

Corre las 8 secciones del pipeline en orden, imprime un resumen en consola y guarda todo en
`docs/figuras/` y `docs/tablas/`. No recibe argumentos. Fija la semilla al inicio
(`set_global_seed`).

## 11. Preguntas generales que puede hacer el profesor

- **"Explícame la diferencia entre Calmar IS y OOS y qué significa la WFE."** Calmar_IS es el
  desempeño DENTRO de la ventana de entrenamiento con los parámetros que el optimizador eligió
  para esa ventana; Calmar_OOS es el desempeño de ESOS MISMOS parámetros (sin reoptimizar) en la
  semana siguiente, nunca vista durante la optimización. WFE = OOS/IS mide qué proporción de la
  ventaja encontrada en entrenamiento sobrevive fuera de muestra.
- **"¿Dónde garantizas que no hay look-ahead?"** Tres lugares: (1) `signals.py` usa solo
  `rolling`/`ewm` hacia atrás y `shift_signal_for_execution` desplaza la señal un lugar; (2)
  `regimes.py` ajusta el escalador/K-means solo con datos hasta el fin de la ventana de
  entrenamiento; (3) el test `test_causalidad_indicadores_en_tres_puntos` (y el equivalente para
  régimen) verifica programáticamente que recalcular sobre un prefijo da el mismo valor en t.
- **"¿Por qué el motor es tan lento/rápido?"** Es puro Python/pandas: cada backtest opera sobre
  arreglos de NumPy con un bucle `for` explícito, pero el CONTEXTO que se le pasa está acotado
  (no recalcula indicadores sobre los 9 meses completos de un episodio, solo sobre la ventana
  más un margen de calentamiento), lo que mantiene cada prueba de Optuna en milisegundos.
- **"¿Qué pasa si el equipo no puede explicar una línea?"** Por diseño, cada función tiene una
  sola responsabilidad y un docstring que explica el supuesto, precisamente para que cualquiera
  de los dos pueda ubicarla y explicarla sin ambigüedad.

## 12. Propuesta de reparto de las 12 diapositivas

| # | Tema | Expositor |
|---|---|---|
| Portada | (no cuenta) | — |
| 1 | Objetivo y resumen ejecutivo (cifras principales) | A |
| 2 | Datos: auditoría y limpieza causal | A |
| 3 | Estrategia: indicadores y fórmula de confirmación 2 de 3 | A |
| 4 | Motor de backtesting: supuestos clave (SL/TP, sin apalancamiento, simetría largo/corto) | A |
| 5 | Walk-forward y optimización híbrida (Random+TPE, restricción de operaciones mínimas) | A |
| 6 | Resultados: portafolio train OOS vs benchmark | B |
| 7 | WFE y degradación train→test (pregunta de análisis 2) | B |
| 8 | Detección de régimen: metodología y validación (persistencia, silhouette) | B |
| 9 | Desempeño por régimen (pregunta de análisis 5) | B |
| 10 | Robustez: sensibilidad ±20% y costo de equilibrio (preguntas 3 y 4) | B |
| 11 | Advertencia de interpretación: fricción de ejecución | B |
| 12 | Limitaciones y conclusiones (pregunta 7) | A y B (cierre conjunto) |
| Cierre | (no cuenta) | — |

A explica más la construcción (datos, estrategia, motor, optimización); B explica más los
resultados (desempeño, régimen, robustez). Ambos deben poder responder preguntas sobre
cualquier diapositiva, no solo las propias.
