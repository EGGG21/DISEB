"""DISEB - Entrenamiento y experimentacion de modelos (Entrega 3).

Modelos: DummyClassifier (referencia), Random Forest (principal) y Regresion Logistica (comparativo).
Cada experimento se corre sobre DOS conjuntos de variables:
    - contrato: las 4 variables del contrato de inferencia de la Entrega 2.
    - ampliado: contrato + variables extra de telemetria/historial (sin variables que filtren el objetivo).

Ejecucion (desde la raiz del repositorio):
    python src/models/train.py --data data/processed/dataset_limpio.csv --output models/
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.preprocess import CONJUNTOS, TARGET, construir_preprocesador  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
REPORTS = ROOT / "reports"

# (id, nombre, familia, constructor del modelo, hiperparametros para documentar)
EXPERIMENTOS = [
    ("exp1_dummy", "DummyClassifier", "Baseline",
     lambda: DummyClassifier(strategy="most_frequent"),
     {"strategy": "most_frequent"}),
    ("exp2_rf_base", "Random Forest", "Modelo principal",
     lambda: RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1,
                                    class_weight=None, random_state=SEED, n_jobs=-1),
     {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "class_weight": None}),
    ("exp3_rf_regularizado", "Random Forest", "Ajuste A",
     lambda: RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                    class_weight="balanced", random_state=SEED, n_jobs=-1),
     {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"}),
    ("exp4_lr_base", "Logistic Regression", "Modelo alternativo",
     lambda: LogisticRegression(C=1.0, penalty="l2", class_weight=None, max_iter=2000),
     {"C": 1.0, "penalty": "l2", "class_weight": None, "max_iter": 2000}),
    ("exp5_lr_balanceado", "Logistic Regression", "Ajuste B",
     lambda: LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", max_iter=2000),
     {"C": 0.1, "penalty": "l2", "class_weight": "balanced", "max_iter": 2000}),
]


def diagnostico(f1_tr, f1_te, es_dummy):
    """Regla simple: brecha Train-Test > 0.10 => sobreajuste; F1 train bajo => subajuste."""
    if es_dummy:
        return "Referencia ingenua (no aprende patrones)"
    brecha = f1_tr - f1_te
    if brecha > 0.10:
        return f"Sobreajuste (brecha F1 Train-Test = {brecha:.3f})"
    if f1_tr < 0.70:
        return f"Subajuste (F1 Train = {f1_tr:.3f}, brecha = {brecha:.3f})"
    return f"Optimo (brecha F1 Train-Test = {brecha:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset_limpio.csv")
    ap.add_argument("--output", default="models/")
    ap.add_argument("--no-stratify", action="store_true", help="particion sin estratificar")
    args = ap.parse_args()

    out = (ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    log_f = open(REPORTS / "entrenamiento.log", "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    data_path = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    log(f"[INFO] Cargando dataset: {args.data}")
    df = pd.read_csv(data_path)
    y = df[TARGET].astype(int)
    log(f"[INFO] {len(df):,} registros | Fallas (1): {y.mean()*100:.2f}% | Sanas (0): {(1-y.mean())*100:.2f}%")

    idx_tr, idx_te = train_test_split(
        df.index, test_size=0.20, random_state=SEED, stratify=None if args.no_stratify else y)
    y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]
    log(f"[INFO] Conjunto Train: {len(idx_tr):,} muestras | Test: {len(idx_te):,} muestras "
        f"(80/20, random_state={SEED}, estratificado={not args.no_stratify})")

    resultados, modelos = [], {}
    for conj, (num, cat) in CONJUNTOS.items():
        feats = num + cat
        X_tr, X_te = df.loc[idx_tr, feats], df.loc[idx_te, feats]
        log(f"\n[INFO] ===== Conjunto de variables: {conj.upper()} ({len(feats)} variables) =====")
        log(f"[INFO] Variables: {', '.join(feats)}")
        for eid, nombre, rol, build, hp in EXPERIMENTOS:
            pipe = Pipeline([("prep", construir_preprocesador(num, cat)), ("modelo", build())])
            log(f"[INFO] Entrenando {eid} [{conj}]: {nombre} {hp} ...")
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            t_train = time.perf_counter() - t0
            t0 = time.perf_counter()
            p_te = pipe.predict(X_te)
            t_inf = time.perf_counter() - t0
            p_tr = pipe.predict(X_tr)
            m = {
                "train": {"accuracy": accuracy_score(y_tr, p_tr), "f1_macro": f1_score(y_tr, p_tr, average="macro")},
                "test": {"accuracy": accuracy_score(y_te, p_te), "f1_macro": f1_score(y_te, p_te, average="macro"),
                         "f1_clase_falla": f1_score(y_te, p_te, pos_label=1),
                         "recall_clase_falla": recall_score(y_te, p_te, pos_label=1, zero_division=0)},
            }
            diag = diagnostico(m["train"]["f1_macro"], m["test"]["f1_macro"], nombre == "DummyClassifier")
            cm = confusion_matrix(y_te, p_te, labels=[0, 1])
            reg = {
                "id": f"{eid}__{conj}", "experimento": eid, "conjunto_variables": conj, "rol": rol,
                "modelo": nombre, "hiperparametros": hp,
                "metricas_train": m["train"], "metricas_test": m["test"],
                "tiempos": {"entrenamiento_s": t_train, "inferencia_total_s": t_inf,
                            "inferencia_ms_por_muestra": t_inf / len(X_te) * 1000},
                "diagnostico": diag,
                "reporte_clasificacion_test": classification_report(
                    y_te, p_te, labels=[0, 1], output_dict=True, zero_division=0),
                "matriz_confusion_test": cm.tolist(),
            }
            resultados.append(reg)
            modelos[reg["id"]] = (pipe, feats, num, cat)
            log(f"       Train F1m={m['train']['f1_macro']:.4f} Acc={m['train']['accuracy']:.4f} | "
                f"Test F1m={m['test']['f1_macro']:.4f} Acc={m['test']['accuracy']:.4f} | "
                f"{t_train:.2f}s / {reg['tiempos']['inferencia_ms_por_muestra']:.4f} ms/muestra | {diag}")
            joblib.dump(pipe, out / f"{eid}__{conj}.joblib")

    # Ganador: mayor F1 Macro en Test, sin contar el Dummy
    candidatos = [r for r in resultados if r["modelo"] != "DummyClassifier"]
    ganador = max(candidatos, key=lambda r: r["metricas_test"]["f1_macro"])
    pipe_g, feats_g, num_g, cat_g = modelos[ganador["id"]]
    log(f"\n[RESULTADO] Modelo ganador: {ganador['id']} ({ganador['modelo']}, variables: {ganador['conjunto_variables']})")
    log(f"[RESULTADO] F1 Macro Test = {ganador['metricas_test']['f1_macro']:.4f} | "
        f"Accuracy Test = {ganador['metricas_test']['accuracy']:.4f} | Meta Entrega 2: >= 0.85")
    Xg_te = df.loc[idx_te, feats_g]
    log("\n[REPORTE DE CLASIFICACION - TEST - GANADOR]\n" + classification_report(
        y_te, pipe_g.predict(Xg_te), labels=[0, 1], target_names=["0 Sana", "1 Falla"], zero_division=0))

    joblib.dump(pipe_g, out / "modelo_ganador.joblib")
    prep = pipe_g.named_steps["prep"]
    joblib.dump(prep.named_transformers_["num"].named_steps["imputer"], out / "imputer.joblib")
    joblib.dump(prep.named_transformers_["num"].named_steps["scaler"], out / "scaler.joblib")
    meta = {"ganador": ganador["id"], "modelo": ganador["modelo"], "conjunto": ganador["conjunto_variables"],
            "features_numericas": num_g, "features_categoricas": cat_g,
            "etiquetas": {"0": "Bateria Sana", "1": "Riesgo Alto de Falla"}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[SUCCESS] Modelo ganador serializado: {out / 'modelo_ganador.joblib'}")
    log(f"[SUCCESS] Imputador y escalador guardados: {out / 'imputer.joblib'}, {out / 'scaler.joblib'}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(np.array(ganador["matriz_confusion_test"]),
                           display_labels=["0 Sana", "1 Falla"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusion (Test)\n{ganador['modelo']} - variables {ganador['conjunto_variables']}")
    fig.tight_layout()
    fig.savefig(REPORTS / "matriz_confusion_test.png", dpi=150)
    plt.close(fig)

    (REPORTS / "resultados.json").write_text(json.dumps(
        {"semilla": SEED, "estratificado": not args.no_stratify, "n_train": len(idx_tr), "n_test": len(idx_te),
         "ganador": ganador["id"], "experimentos": resultados}, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[SUCCESS] Reportes guardados en {REPORTS}")
    log_f.close()


if __name__ == "__main__":
    main()
