"""DISEB - Preprocesamiento de datos (compartido por entrenamiento e inferencia).

Uso para generar el dataset limpio desde el CSV original:
    python src/data/preprocess.py
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "ev_battery_failure prediction Dataset.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "dataset_limpio.csv"

TARGET = "battery_failure"

# Variables del contrato de inferencia de la Entrega 2
FEATURES_CONTRATO_NUM = ["vehicle_age_years", "cell_voltage_avg", "cell_temperature_avg"]
FEATURES_CONTRATO_CAT = ["battery_chemistry"]

# Variables adicionales de telemetria/historial (conjunto "ampliado"). Se EXCLUYEN a proposito
# las columnas que ya son un resultado del deterioro (state_of_health, battery_health_percent,
# capacity_loss_percent, thermal_runaway_risk, predicted_remaining_life_cycles, etc.)
# para evitar fuga de informacion (data leakage).
FEATURES_EXTRA_NUM = [
    "previous_faults", "BMS_warning_count", "abnormal_voltage_events",
    "internal_resistance", "cell_temperature_max", "voltage_imbalance",
    "temperature_variance", "charge_efficiency", "discharge_efficiency",
    "cycle_count", "sensor_fault_count", "cooling_system_health",
    "maintenance_score", "odometer_km",
]

CONJUNTOS = {
    "contrato": (FEATURES_CONTRATO_NUM, FEATURES_CONTRATO_CAT),
    "ampliado": (FEATURES_CONTRATO_NUM + FEATURES_EXTRA_NUM, FEATURES_CONTRATO_CAT),
}

# Rangos plausibles para validar entradas en inferencia. Cubren los valores reales del CSV
# (la Entrega 2 documenta 3.0-4.5 V y -10 a 60 C, pero el CSV llega a -14.2 y 65.1 C).
RANGES = {
    "vehicle_age_years": (0.0, 30.0),
    "cell_voltage_avg": (2.5, 4.5),
    "cell_temperature_avg": (-30.0, 80.0),
}
QUIMICAS_VALIDAS = ["NMC", "LFP", "LMO", "NCA", "LTO"]


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza basica: duplicados, objetivo valido, columnas del proyecto y tipos numericos."""
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    columnas = [TARGET] + FEATURES_CONTRATO_NUM + FEATURES_CONTRATO_CAT + FEATURES_EXTRA_NUM
    df = df[columnas].copy()
    for c in FEATURES_CONTRATO_NUM + FEATURES_EXTRA_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def construir_preprocesador(num_cols, cat_cols) -> ColumnTransformer:
    """Mediana + StandardScaler para numericas; moda + One-Hot para categoricas."""
    pasos = [("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())]), list(num_cols))]
    if cat_cols:
        pasos.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                      list(cat_cols)))
    return ColumnTransformer(pasos)


def main():
    parser = argparse.ArgumentParser(description="Genera data/processed/dataset_limpio.csv")
    parser.add_argument("--raw", default=str(RAW_PATH))
    parser.add_argument("--out", default=str(CLEAN_PATH))
    args = parser.parse_args()
    df = pd.read_csv(args.raw)
    limpio = limpiar(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    limpio.to_csv(args.out, index=False)
    print(f"[INFO] Original: {len(df)} filas x {df.shape[1]} columnas")
    print(f"[SUCCESS] Limpio: {len(limpio)} filas x {limpio.shape[1]} columnas -> {args.out}")


if __name__ == "__main__":
    main()
