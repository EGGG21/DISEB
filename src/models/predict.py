"""DISEB - Funcion de inferencia desacoplada: predecir(datos).

Recibe un diccionario con lecturas de una bateria, valida la entrada, aplica el mismo
preprocesamiento del entrenamiento (guardado dentro del pipeline ganador) y devuelve el diagnostico.

Entrada minima (contrato de la Entrega 2):
    {"vehicle_age_years": 5.2, "cell_voltage_avg": 3.65,
     "cell_temperature_avg": 25.4, "battery_chemistry": "NMC"}
Las variables extra del modelo ganador son opcionales: si faltan, se imputan con la mediana del entrenamiento.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.preprocess import QUIMICAS_VALIDAS, RANGES  # noqa: E402

MODELS_DIR = ROOT / "models"
REQUERIDAS = ["vehicle_age_years", "cell_voltage_avg", "cell_temperature_avg", "battery_chemistry"]
_cache = {}


def _cargar():
    """Carga el modelo y sus metadatos una sola vez por proceso."""
    if not _cache:
        _cache["modelo"] = joblib.load(MODELS_DIR / "modelo_ganador.joblib")
        _cache["meta"] = json.loads((MODELS_DIR / "meta.json").read_text(encoding="utf-8"))
    return _cache["modelo"], _cache["meta"]


def _validar(datos):
    """Devuelve un mensaje de error (str) o None si la entrada es valida."""
    if not isinstance(datos, dict):
        return "La entrada debe ser un diccionario."
    faltan = [c for c in REQUERIDAS if datos.get(c) in (None, "")]
    if faltan:
        return f"Faltan campos obligatorios: {', '.join(faltan)}."
    for campo, (lo, hi) in RANGES.items():
        try:
            v = float(datos[campo])
        except (TypeError, ValueError):
            return f"El campo '{campo}' debe ser numerico."
        if not np.isfinite(v) or not lo <= v <= hi:
            return f"El campo '{campo}' esta fuera de rango ({lo} a {hi})."
    if str(datos["battery_chemistry"]).upper() not in QUIMICAS_VALIDAS:
        return f"battery_chemistry debe ser una de: {', '.join(QUIMICAS_VALIDAS)}."
    return None


def predecir(datos: dict) -> dict:
    t0 = time.perf_counter()
    error = _validar(datos)
    if error:
        return {"status": "error", "mensaje": error}
    try:
        modelo, meta = _cargar()
        columnas = meta["features_numericas"] + meta["features_categoricas"]
        fila = {c: np.nan for c in columnas}
        for c in columnas:
            if datos.get(c) not in (None, ""):
                fila[c] = datos[c]
        fila["battery_chemistry"] = str(datos["battery_chemistry"]).upper()
        X = pd.DataFrame([fila], columns=columnas)
        for c in meta["features_numericas"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        clase = int(modelo.predict(X)[0])
        p_falla = float(modelo.predict_proba(X)[0][1]) if hasattr(modelo, "predict_proba") else float(clase)
        return {
            "prediccion": clase,
            "clase_etiqueta": "Riesgo Alto de Falla" if clase == 1 else "Bateria Sana",
            "confianza_o_score": round(p_falla if clase == 1 else 1 - p_falla, 4),
            "probabilidad_falla": round(p_falla, 4),
            "tiempo_ms": round((time.perf_counter() - t0) * 1000, 2),
            "status": "success",
        }
    except Exception as e:  # el dashboard no debe romperse
        return {"status": "error", "mensaje": f"Error interno de inferencia: {e}"}


if __name__ == "__main__":
    ejemplo = {"vehicle_age_years": 5.2, "cell_voltage_avg": 3.65,
               "cell_temperature_avg": 25.4, "battery_chemistry": "NMC"}
    print("Entrada valida  ->", predecir(ejemplo))
    print("Voltaje invalido ->", predecir({**ejemplo, "cell_voltage_avg": 9.9}))
    print("Campo faltante   ->", predecir({"cell_voltage_avg": 3.6}))
    ejemplo_riesgo = {**ejemplo, "cell_temperature_avg": 58.0, "cell_voltage_avg": 3.1, "vehicle_age_years": 10.5,
                      "previous_faults": 4, "BMS_warning_count": 8, "abnormal_voltage_events": 6}
    print("Caso de riesgo   ->", predecir(ejemplo_riesgo))
