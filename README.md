# Laboratorio 02 — Estrategias de Trading con Análisis Técnico (Nivel B)

**Microestructuras y Sistemas de Trading — ITESO**

## Integrantes

- Ernesto Andrés González Lomelí
- Rogelio Adrián Arroyo Valencia

## Nivel de alcance

**Nivel B** (equipo de 2 integrantes): Nivel A completo (estrategia multi-indicador con
confirmación 2 de 3, motor de backtesting event-driven, optimización walk-forward sobre
`btc_project_train.csv` y evaluación en `btc_project_test.csv`) **más** detección dinámica
de régimen de mercado (K-means sobre una matriz de variables de volatilidad, tendencia y
reversión, con parámetros optimizados por régimen dentro de cada ventana de entrenamiento).

## Descripción del proyecto

El proyecto implementa una estrategia sistemática sobre BTCUSDT de 5 minutos que combina
tres indicadores técnicos de familias distintas (cruce de medias exponenciales para
tendencia, RSI para momento y ruptura de Bandas de Bollinger para volatilidad) bajo una
regla de confirmación de "2 de 3" para decidir la dirección de cada posición. Las señales se
ejecutan con una barra de rezago estricto (decisión en el cierre de la barra t, ejecución en la
apertura de t+1) dentro de un motor de backtesting event-driven que modela comisiones,
stop-loss y take-profit basados en ATR, y la imposibilidad de apalancamiento tanto en largos
como en cortos. Un detector de régimen por K-means clasifica cada barra en tendencia,
reversión a la media o crisis usando una ventana móvil de una semana, y cada régimen tiene su
propio conjunto de hiperparámetros, optimizado con un esquema híbrido Random Search + Optuna
(TPE) dentro de un walk-forward de un mes de entrenamiento y una semana de prueba sobre el
conjunto de entrenamiento. Los parámetros y el modelo de régimen de la última ventana se
congelan y se evalúan una sola vez, sin reoptimizar, sobre el conjunto de prueba.

## Instalación

Requiere **Python 3.11** (el proyecto se desarrolló y probó con Python 3.11.9).

```bash
cd Lab02_MyST_EquipoN
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Cómo reproducir todos los resultados

Un solo comando, sin argumentos, desde la raíz de `Lab02_MyST_EquipoN/`:

```bash
python main.py
```

Esto ejecuta, en orden: auditoría y limpieza de datos, diagnóstico de costo de ida y vuelta
contra ATR, walk-forward de optimización CON régimen sobre train, walk-forward de referencia
SIN régimen (misma función objetivo y presupuesto, usado para la pregunta de análisis 5),
congelamiento de parámetros, evaluación en test, cálculo de métricas (con benchmark de comprar
y mantener encadenado por episodios) y validación de régimen, análisis de robustez, y
generación de todas las figuras y tablas en `docs/figuras/` y `docs/tablas/`.

Para correr las pruebas automatizadas:

```bash
pytest tests/ -v
```

## Semilla aleatoria

Se fija una sola vez, en `src/config.py` (`RANDOM_SEED = 42`), y se propaga explícitamente a
todo lo aleatorio del proyecto: el muestreador aleatorio y el muestreador TPE de Optuna (con
una semilla derivada determinísticamente por ventana y por régimen, `seed + w_idx*10 + r`,
para que cada combinación tenga su propio flujo de números aleatorios sin perder
reproducibilidad), el `KMeans` del detector de régimen, y el bootstrap de la prueba de
significancia entre regímenes. Dos ejecuciones de `python main.py` producen exactamente los
mismos números.

## Tiempo aproximado de ejecución

Entre 18 y 20 minutos en una laptop de oficina (sin GPU); se midieron 18.8 y 19.3 minutos en dos
ejecuciones completas de la versión corregida (que agrega, sobre la versión anterior, una segunda
pasada de walk-forward SIN régimen como referencia — sección "Walk-forward" más abajo). El tiempo
está dominado por las 76 ventanas generadas por `generate_walkforward_windows`: la versión CON
régimen procesa 74 (2 se omiten porque esa ventana no alcanza el mínimo de filas con variables de
régimen completas para ajustar K-means) y la versión SIN régimen procesa las 76 (no depende de
ajustar ningún modelo de régimen). Entre ambas suman 35,520 configuraciones evaluadas. El motor de
backtesting es puro Python/pandas: no requiere compilación ni dependencias binarias adicionales.

**Determinismo verificado:** se corrió `python main.py` dos veces con el mismo código y se
compararon los archivos de salida. `metricas_resumen.csv`, `walkforward_ventanas.csv`,
`walkforward_baseline_ventanas.csv` y `sensibilidad_costos.csv` resultaron **idénticos byte a
byte**. El único archivo con una diferencia fue `parametros_congelados.json`, y solo en los
últimos 1-2 dígitos significativos de los centroides de K-means (orden de 1e-13 relativo): ruido
de punto flotante típico de las
rutinas de álgebra lineal (BLAS) bajo distintos hilos del sistema operativo, no una falla de la
semilla. No cambia ningún parámetro elegido, ninguna métrica ni ninguna cifra reportada.

## Uso de asistencia de IA

Este proyecto se desarrolló con asistencia de Claude (Anthropic) como asistente de
programación, dentro de las reglas del curso (sección 6 del Lab02.pdf). La asistencia de IA se
usó para: redactar la estructura inicial de los módulos de `src/` a partir de las decisiones de
diseño discutidas con el equipo (limpieza causal de datos, convenciones del motor de
backtesting, esquema de optimización híbrido, matriz de features de régimen); generar las
pruebas de `tests/`; y redactar el borrador del reporte y la presentación a partir de los
resultados numéricos ya calculados por el pipeline. Una revisión externa posterior señaló
defectos de lógica y de análisis en esa primera versión (función objetivo, espacio de búsqueda
de take-profit, un adelanto de una barra en el cierre por cambio de régimen, la métrica de
diferenciación por régimen, la ausencia de una corrida de referencia sin régimen y el benchmark
de test); la corrección de cada defecto, las pruebas nuevas que los verifican y la
reescritura del reporte con las cifras corregidas también se hicieron con asistencia de IA,
siguiendo el mismo esquema: cada corrección se motivó con diagnósticos del propio proyecto (no
con el resultado de test) y quedó documentada en `docs/reporte.pdf`. Cada integrante revisó,
ejecutó y puede explicar la totalidad del código entregado, incluidas las correcciones; ningún
resultado numérico del reporte, la presentación o este README fue escrito a mano: todos
provienen de una ejecución real de `python main.py`.

## Estructura del repositorio

```
Lab02_MyST_EquipoN/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/                  # btc_project_train.csv y btc_project_test.csv, sin modificar
├── src/
│   ├── config.py          # constantes, semilla y supuestos globales
│   ├── data.py             # carga, auditoría y limpieza causal
│   ├── signals.py          # indicadores y regla de confirmación 2 de 3
│   ├── backtest.py         # motor event-driven con costos
│   ├── metrics.py          # Sharpe, Sortino, Calmar, MDD, Win Rate, fricción
│   ├── optimize.py         # walk-forward, Optuna híbrido, robustez
│   ├── regimes.py          # detección de régimen (K-means)
│   └── plots.py            # todas las figuras
├── tests/                 # pruebas de pytest (causalidad, contabilidad, régimen, etc.)
├── notebooks/
│   └── analysis.ipynb      # solo análisis y figuras, importa de src/
└── docs/
    ├── reporte.pdf
    ├── presentacion.pdf
    ├── figuras/             # generadas por main.py
    └── tablas/              # generadas por main.py
```
