# Propuesta de reparto de archivos y commits

Solo es una propuesta: no se ejecutó ningún comando de git. Divide el trabajo en dos grupos
equilibrados (uno por integrante) y en un orden lógico de subida, respetando las dependencias
del proyecto (config y datos antes que la estrategia, la estrategia antes que el motor, el motor
antes que la optimización, todo antes que `main.py` y los documentos finales).

Antes de empezar, ambos integrantes deben tener el repositorio en GitHub creado (vacío) y
clonado localmente, con su propio usuario de git configurado (`git config user.name` /
`user.email`) para que los commits queden atribuidos correctamente.

## Commits de Ernesto Andrés González Lomelí

| # | Archivos | Mensaje de commit sugerido |
|---|---|---|
| 1 | `README.md`, `requirements.txt`, `.gitignore`, carpetas vacías `docs/`, `notebooks/`, `tests/` (con un `.gitkeep` si hace falta) | `Agregar estructura base del proyecto y documentacion inicial` |
| 2 | `data/btc_project_train.csv`, `data/btc_project_test.csv`, `src/__init__.py`, `src/config.py` | `Agregar datos crudos congelados y configuracion global del proyecto` |
| 3 | `src/data.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_data.py` | `Implementar carga, auditoria y limpieza causal de los datos` |
| 4 | `src/metrics.py` | `Implementar metricas de desempeno y estimacion de friccion de ejecucion` |
| 5 | `src/regimes.py`, `tests/test_regimes.py` | `Implementar deteccion de regimen de mercado con K-means` |
| 6 | `main.py` | `Agregar el orquestador principal del pipeline (main.py)` |

## Commits de Rogelio Adrián Arroyo Valencia

| # | Archivos | Mensaje de commit sugerido |
|---|---|---|
| 1 | `src/signals.py`, `tests/test_signals.py` | `Implementar indicadores tecnicos y regla de confirmacion 2 de 3` |
| 2 | `src/backtest.py`, `tests/test_backtest.py` | `Implementar motor de backtesting event-driven con costos` |
| 3 | `src/optimize.py`, `tests/test_optimize.py` | `Implementar walk-forward, optimizacion hibrida y analisis de robustez` |
| 4 | `src/plots.py` | `Implementar las figuras obligatorias y de diagnostico` |
| 5 | `notebooks/analysis.ipynb` | `Agregar notebook de analisis exploratorio` |
| 6 | `docs/figuras/*`, `docs/tablas/*` (generados por la corrida final de `python main.py`) | `Agregar figuras y tablas de la corrida final del pipeline` |
| 7 | `docs/reporte.pdf`, `docs/presentacion.pdf` | `Agregar reporte ejecutivo y presentacion final` |

## Orden lógico de subida (intercalado)

1. Ernesto #1 (estructura base)
2. Ernesto #2 (datos y config)
3. Ernesto #3 (data.py)
4. Rogelio #1 (signals.py) — depende de config.py, ya disponible
5. Rogelio #2 (backtest.py) — depende de signals.py
6. Ernesto #4 (metrics.py) — independiente de signals/backtest, puede ir en paralelo
7. Ernesto #5 (regimes.py) — depende de config.py
8. Rogelio #3 (optimize.py) — depende de signals.py, backtest.py, metrics.py y regimes.py: va al final de la lógica central
9. Rogelio #4 (plots.py) — depende de metrics.py/optimize.py para los tipos de datos que grafica
10. Ernesto #6 (main.py) — depende de todo lo anterior
11. Rogelio #5 (notebook) — depende de que main.py ya exista y se haya corrido al menos una vez
12. Rogelio #6 (figuras y tablas) — resultado de correr `python main.py`
13. Rogelio #7 (reporte y presentación) — se redactan con los números de las figuras/tablas ya generadas

Quedan 6 commits para Ernesto y 7 para Rogelio: un reparto equilibrado (13 commits en total),
con cada módulo subido por la persona que lo implementó y lo puede explicar a detalle.
