# ENTREGA 3 — DISEB: textos con resultados reales
(Todos los números salen de `reports/resultados.json` y `reports/entrenamiento.log`, generados al ejecutar el código con el CSV real: 20,000 filas; Train 16,000 / Test 4,000; random_state=42.)

## Sección 3 — Matriz de experimentación
### Conjunto AMPLIADO (18 variables: las 4 del contrato + 14 de telemetría/historial) — conjunto del modelo ganador
| Variante | Modelo | Hiperparámetros clave | Métrica principal (Test) | Métrica secundaria (Test) | Tiempo entrenamiento / inferencia por muestra | Diagnóstico |
|---|---|---|---|---|---|---|
| Exp 1: Baseline | DummyClassifier | strategy=most_frequent | F1 Macro: 0.4821 | Accuracy: 0.9307 | 0.04 s / 0.0015 ms | Train F1m 0.4821 (Acc 0.9308) vs Test F1m 0.4821. Referencia ingenua (no aprende patrones) |
| Exp 2: Modelo principal | Random Forest | n_estimators=200, max_depth=None, min_samples_leaf=1, class_weight=None | F1 Macro: 0.7360 | Accuracy: 0.9493 | 5.38 s / 0.0214 ms | Train F1m 1.0000 (Acc 1.0000) vs Test F1m 0.7360. Sobreajuste (brecha F1 Train-Test = 0.264) |
| Exp 3: Ajuste A | Random Forest | n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight=balanced | F1 Macro: 0.7847 | Accuracy: 0.9270 | 5.67 s / 0.0323 ms | Train F1m 0.8620 (Acc 0.9537) vs Test F1m 0.7847. Optimo (brecha F1 Train-Test = 0.077) |
| Exp 4: Modelo alternativo | Logistic Regression | C=1.0, penalty=l2, class_weight=None, max_iter=2000 | F1 Macro: 0.8116 | Accuracy: 0.9563 | 0.06 s / 0.0015 ms | Train F1m 0.8240 (Acc 0.9598) vs Test F1m 0.8116. Optimo (brecha F1 Train-Test = 0.012) |
| Exp 5: Ajuste B | Logistic Regression | C=0.1, penalty=l2, class_weight=balanced, max_iter=2000 | F1 Macro: 0.7542 | Accuracy: 0.9005 | 0.07 s / 0.0015 ms | Train F1m 0.7666 (Acc 0.9073) vs Test F1m 0.7542. Optimo (brecha F1 Train-Test = 0.012) |

### Conjunto CONTRATO (solo las 4 variables de la Entrega 2: vehicle_age_years, cell_voltage_avg, cell_temperature_avg, battery_chemistry)
| Variante | Modelo | Hiperparámetros clave | Métrica principal (Test) | Métrica secundaria (Test) | Tiempo entrenamiento / inferencia por muestra | Diagnóstico |
|---|---|---|---|---|---|---|
| Exp 1: Baseline | DummyClassifier | strategy=most_frequent | F1 Macro: 0.4821 | Accuracy: 0.9307 | 0.02 s / 0.0013 ms | Train F1m 0.4821 (Acc 0.9308) vs Test F1m 0.4821. Referencia ingenua (no aprende patrones) |
| Exp 2: Modelo principal | Random Forest | n_estimators=200, max_depth=None, min_samples_leaf=1, class_weight=None | F1 Macro: 0.5920 | Accuracy: 0.9295 | 2.46 s / 0.0245 ms | Train F1m 1.0000 (Acc 1.0000) vs Test F1m 0.5920. Sobreajuste (brecha F1 Train-Test = 0.408) |
| Exp 3: Ajuste A | Random Forest | n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight=balanced | F1 Macro: 0.6159 | Accuracy: 0.8207 | 2.88 s / 0.0331 ms | Train F1m 0.7097 (Acc 0.8650) vs Test F1m 0.6159. Optimo (brecha F1 Train-Test = 0.094) |
| Exp 4: Modelo alternativo | Logistic Regression | C=1.0, penalty=l2, class_weight=None, max_iter=2000 | F1 Macro: 0.5719 | Accuracy: 0.9315 | 0.03 s / 0.0018 ms | Train F1m 0.5921 (Acc 0.9346) vs Test F1m 0.5719. Subajuste (F1 Train = 0.592, brecha = 0.020) |
| Exp 5: Ajuste B | Logistic Regression | C=0.1, penalty=l2, class_weight=balanced, max_iter=2000 | F1 Macro: 0.5714 | Accuracy: 0.7500 | 0.04 s / 0.0015 ms | Train F1m 0.5841 (Acc 0.7622) vs Test F1m 0.5714. Subajuste (F1 Train = 0.584, brecha = 0.013) |

## Análisis crítico
**1. Modelo ganador.** El ganador fue **Regresión Logística (C=1.0, L2) con el conjunto ampliado**, con F1 Macro = 0.8116 y Accuracy = 0.9563 en Test. El mejor Random Forest (regularizado, max_depth=12, min_samples_leaf=5, class_weight balanced) obtuvo F1 Macro = 0.7847 y Accuracy = 0.9270, una diferencia de 0.0269 puntos de F1 Macro. Una explicación posible (no verificada con más pruebas) es que, con señales como previous_faults, BMS_warning_count y abnormal_voltage_events, la relación con la falla es en buena parte lineal, y el ensamble tiende a memorizar a la clase minoritaria (~6.9 % de las filas), por lo que el modelo lineal generaliza mejor. En la clase Falla, el ganador tiene precisión 0.73, recall 0.58 y F1 0.65.

**Meta de la Entrega 2 (F1 Macro ≥ 0.85): NO se cumple** (mejor resultado: 0.8116). Con solo las 4 variables del contrato, el mejor F1 Macro fue 0.6159: esas variables por sí solas casi no separan las clases. Líneas de mejora para la Entrega 4: ajustar el umbral de decisión con validación cruzada dentro de Train, probar Gradient Boosting y revisar con el responsable de Datos qué variables extra se pueden pedir en el dashboard.

**2. Sobreajuste.** Ganador: Accuracy Train 0.9598 / Test 0.9563; F1 Macro Train 0.8240 / Test 0.8116 (brecha 0.0124). El Random Forest base (Exp 2) sí sobreajustó: F1 Macro Train 1.0000 vs Test 0.7360. Al limitar la profundidad (max_depth=12), aumentar min_samples_leaf=5 y usar class_weight="balanced" (Exp 3), la brecha bajó a 0.0773 y el F1 Macro de Test subió a 0.7847. En Regresión Logística, bajar C a 0.1 con pesos balanceados (Exp 5) mantuvo la brecha en ~0.01 pero redujo el F1 Macro de Test a 0.7542: detecta más fallas (recall 0.93) pero genera muchas falsas alarmas.

## Sección 4 — Evidencias
- Log: pega el contenido de `reports/entrenamiento.log`.
- Imagen: inserta `reports/matriz_confusion_test.png`. Matriz del ganador en Test (filas = real, columnas = predicho): Sana→Sana 3665, Sana→Falla 58, Falla→Sana 117, Falla→Falla 160.

## Sección 5 — Ejemplo real de predecir()
Entrada: `{"vehicle_age_years": 5.2, "cell_voltage_avg": 3.65, "cell_temperature_avg": 25.4, "battery_chemistry": "NMC"}`
Salida real: `{'prediccion': 0, 'clase_etiqueta': 'Bateria Sana', 'confianza_o_score': 0.9992, 'probabilidad_falla': 0.0008, 'status': 'success'}`
(el campo tiempo_ms varía entre ejecuciones, ~10–30 ms). Las variables extra son opcionales: si el dashboard no las manda, se imputan con la mediana del entrenamiento.

## Sección 6 — Sugerencia de estado (confírala tú)
- Semáforo sugerido: **CON RETRASO MODERADO** (pipeline, modelos y predict.py funcionan, pero F1 Macro 0.8116 < 0.85).
- Blocker: no se alcanza la meta de F1 Macro; la recuperación de fallas (recall 0.58) es mejorable.

## Supuestos
1. Se tomó la Entrega 2 (clasificación de `battery_failure`) como vigente; la Entrega 1 hablaba de regresión del SoH.
2. La partición 80/20 con random_state=42 se hizo **estratificada** (la Entrega 2 no lo especifica). Si ya tienen un Test Set guardado, ese debe prevalecer (el script acepta `--no-stratify`).
3. El conjunto ampliado excluye a propósito state_of_health, battery_health_percent, capacity_loss_percent, thermal_runaway_risk, predicted_remaining_life_cycles y similares, porque ya reflejan el deterioro (posible fuga de datos).
4. El CSV trae temperaturas de −14.2 a 65.1 °C, fuera del rango −10 a 60 °C de la Entrega 2; por eso `predecir()` valida con rangos algo más amplios.
5. Los campos de rol/equipo y la Sección 6 hay que confirmarlos con tu equipo.
