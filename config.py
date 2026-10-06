"""Constantes y supuestos globales del proyecto.

Toda semilla aleatoria y todo parámetro fijo del laboratorio vive en este
único módulo. Ningún otro archivo debe redefinir estos valores: deben
importarse desde aquí para garantizar reproducibilidad.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Rutas (relativas a la ubicación de este archivo, no al directorio de trabajo)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figuras"
TABLES_DIR = DOCS_DIR / "tablas"
TRAIN_CSV = DATA_DIR / "btc_project_train.csv"
TEST_CSV = DATA_DIR / "btc_project_test.csv"

# ---------------------------------------------------------------------------
# Parámetros fijos del problema (Lab02.pdf, sección 3.1)
# ---------------------------------------------------------------------------
COMMISSION = 0.00125          # 0.125% por operación, a la entrada y a la salida
INITIAL_CAPITAL = 1_000_000.0  # capital inicial en USD
CONFIRMATION_MIN_VOTES = 2     # al menos 2 de 3 indicadores deben coincidir
MAX_LEVERAGE = 1.0             # sin apalancamiento: nocional <= efectivo disponible

# ---------------------------------------------------------------------------
# Rejilla temporal y limpieza de datos (ver auditoría en src/data.py)
# ---------------------------------------------------------------------------
BAR_MINUTES = 5
# Umbral para considerar que dos marcas de tiempo CRUDAS (antes de eliminar
# filas con OHLC vacío) forman un "hueco grande" de calendario: una barra
# completa ausente del archivo, no solo un precio vacío. Con este umbral de
# 30 minutos se detectan exactamente los tres huecos estructurales conocidos
# (≈65 min y ≈48 h en train; ≈122 días en test) y ninguno más: se verificó
# empíricamente que no existen diferencias de marca de tiempo cruda entre 5 y
# 30 minutos en ninguno de los dos archivos.
#
# Las rachas de barras con OHLC vacío (hasta 270 barras = 22.5 h seguidas) son
# un fenómeno distinto: la fila SÍ existe en la rejilla de 5 minutos, solo
# falta el precio. Se interpretan como fallas de reporte del proveedor de
# datos, no como una detención real del mercado, así que NO disparan el
# cierre forzado de posiciones ni el reinicio de calentamiento: solo se
# eliminan (ver REGLA_NAN en data.py). Si el precio hubiera saltado de forma
# relevante durante esa racha, la convención de ejecutar al precio de
# apertura cuando la barra abre más allá del nivel de SL/TP ya evita un
# llenado irreal al reanudarse los datos.
GAP_THRESHOLD_MINUTES = 30

# ---------------------------------------------------------------------------
# Régimen de mercado (Nivel B)
# ---------------------------------------------------------------------------
REGIME_WINDOW_BARS = 7 * 24 * 60 // BAR_MINUTES  # 2016 barras = 1 semana
# Número de barras de calentamiento exigidas al inicio de cada episodio
# (tramo continuo de datos entre dos huecos grandes) antes de permitir
# operar. Se iguala a la ventana de régimen (la más larga de todas las
# ventanas causales del proyecto) para garantizar que tanto los indicadores
# como el modelo de régimen tengan historia suficiente y válida.
WARMUP_BARS = REGIME_WINDOW_BARS
# Frecuencia de actualización de la etiqueta de régimen. Se elige 2 horas
# (24 barras) como punto intermedio en el rango permitido de 1-6 horas:
# suficientemente ágil para capturar el inicio de una crisis de volatilidad,
# pero suficientemente lento para no generar transiciones espurias barra a
# barra (ver validación de persistencia en el reporte).
REGIME_UPDATE_HOURS = 2
REGIME_UPDATE_BARS = REGIME_UPDATE_HOURS * 60 // BAR_MINUTES
N_REGIMES = 3
REGIME_NAMES = ("tendencia", "reversion", "crisis")
# Percentil de volatilidad histórica a partir del cual, en el desempate por
# reglas, un grupo se considera de "crisis" (coherente con la heurística de
# las notas de clase: sigma_t > percentil 0.90).
REGIME_CRISIS_VOL_PERCENTILE = 0.90
# Qué hacer con una posición abierta cuando cambia el régimen entre una barra
# y la siguiente: se cierra de inmediato en la apertura de la barra siguiente.
# Es la opción conservadora: evita sostener un tamaño de posición y un
# SL/TP calibrados para un régimen de volatilidad distinto al vigente.
REGIME_TRANSITION_ACTION = "close"

# ---------------------------------------------------------------------------
# Walk-forward (Lab02.pdf, sección 3.3 y 3.4)
# ---------------------------------------------------------------------------
WALKFORWARD_TRAIN_MONTHS = 1  # etiqueta nominal: "1 mes" de entrenamiento
# Se implementa como 30 días fijos (no con DateOffset(months=1)) porque los
# meses de calendario no tienen duración constante: usar DateOffset generaba
# ventanas de entrenamiento de 28 a 31 días, lo que desalineaba el avance de
# `is_end` respecto del paso fijo de 7 días entre ventanas y producía
# semanas de prueba (OOS) ligeramente traslapadas entre ventanas
# consecutivas (una misma barra podía evaluarse OOS más de una vez). Con un
# tamaño de ventana fijo en días, el avance de `is_end` es siempre idéntico
# al paso entre ventanas y las semanas OOS nunca se traslapan.
WALKFORWARD_TRAIN_DAYS = 30
WALKFORWARD_TEST_DAYS = 7
WALKFORWARD_STEP_DAYS = 7

# Número mínimo de operaciones dentro de una ventana de entrenamiento para
# que una configuración de hiperparámetros se considere válida. Con ventanas
# de 1 mes (~8640 barras de 5 min) separadas en 3 regímenes, un régimen poco
# frecuente puede quedarse con pocos cientos de barras; exigir, por ejemplo,
# 30 operaciones (el valor usado por el profesor en un laboratorio con
# ventanas más largas) eliminaría casi todas las ventanas por régimen. Se usa
# un mínimo más bajo pero aun así informativo: N_MIN_TRADES = 8, que exige al
# menos unas pocas operaciones por semana dentro del mes de entrenamiento y
# sigue permitiendo calcular un Win Rate y un Calmar con algo de significado.
N_MIN_TRADES = 8
PENALTY_VALUE = -1e9

# Presupuesto de pruebas de optimización por régimen y por ventana (entre 100
# y 200 según el PDF). Esquema híbrido "warm-start": primero una fase
# exploratoria de Random Search y después Optuna (TPE) inicializado con los
# mejores puntos de esa fase.
N_TRIALS_RANDOM = 40
N_TRIALS_TPE = 80
N_TRIALS_TOTAL = N_TRIALS_RANDOM + N_TRIALS_TPE  # 120, dentro de [100, 200]

# ---------------------------------------------------------------------------
# Métricas: convenciones de anualización
# ---------------------------------------------------------------------------
BARS_PER_DAY = 24 * 60 // BAR_MINUTES   # 288 barras de 5 min por día
TRADING_DAYS_PER_YEAR = 365              # BTC opera 24/7, 365 días al año
ANNUALIZATION_FACTOR_BARS = BARS_PER_DAY * TRADING_DAYS_PER_YEAR
RISK_FREE_RATE_ANNUAL = 0.0  # tasa libre de riesgo anual asumida, declarada

# ---------------------------------------------------------------------------
# Robustez / sensibilidad
# ---------------------------------------------------------------------------
SENSITIVITY_PCT = 0.20
COST_MULTIPLIERS = (1.0, 2.0, 3.0)

# ---------------------------------------------------------------------------
# Estimación de fricción de ejecución (sección 10 del reporte)
# ---------------------------------------------------------------------------
# Parámetro Y de la ley de la raíz cuadrada de impacto de mercado. Es un
# valor típico citado en la literatura de microestructura para mercados de
# cripto líquidos (orden de magnitud 0.5-1.0); se declara como supuesto.
MARKET_IMPACT_Y = 0.8
# Volumen diario en dólares asumido para BTCUSDT (spot + derivados en los
# principales exchanges), usado solo para la ley de la raíz cuadrada de
# impacto de mercado. Se declara como supuesto externo porque la columna
# Volume del archivo no es confiable (ver auditoría de datos): es un orden
# de magnitud razonable para 2022-2024 (decenas de miles de millones de
# USD/día), no una medición propia del proyecto.
ASSUMED_DAILY_DOLLAR_VOLUME = 15_000_000_000.0
