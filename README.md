# 🚛 Predictive Maintenance – Scania APS Failure Detection

Vorhersage von Ausfällen im **Air Pressure System (APS)** von Scania-LKWs mithilfe von Machine Learning.

---

## 📋 Projektübersicht

Das Air Pressure System (APS) ist ein sicherheitskritisches System in schweren LKWs — es steuert Bremsen und Gangwechsel. Dieses Projekt entwickelt ein ML-Modell, das APS-Ausfälle **frühzeitig erkennt**, bevor es zu kostspieligen Pannen kommt.

| Eigenschaft | Details |
|---|---|
| **Datensatz** | Scania APS Failure (UCI / IDA Challenge 2016) |
| **Trainingsset** | 60.000 LKWs (59.000 in Ordnung, 1.000 defekt) |
| **Features** | 170 anonymisierte Sensoren |
| **Zielvariable** | `pos` = APS defekt / `neg` = APS in Ordnung |

---

## 🎯 Ergebnis

Das finale Modell (Gaussian Naive Bayes) auf den **Testdaten**:

| Metrik | Wert |
|---|---|
| **Recall (defekte LKWs erkannt)** | **91 %** |
| Accuracy | 87 % |
| Precision (defekte Klasse) | 11 % |

> ⚠️ Bei diesem Problem hat **Recall oberste Priorität**: Ein übersehener defekter LKW (Cost = 500) ist deutlich teurer als ein Fehlalarm (Cost = 10).

---

## 🔄 Methodik

### 1. Explorative Datenanalyse (EDA)
- Analyse von 171 Features auf fehlende Werte, Konstanz und Korrelation
- **70 Features entfernt**: 28 mit >10% Fehlwerten, 1 konstant, 35 hochkorreliert (>0.9), 6 nahezu konstant
- Ergebnis: 100 relevante Features

### 2. Preprocessing Pipeline (custom sklearn)

```python
Pipeline([
    ('selector', FeatureDropper()),     # Feature-Selektion
    ('outlier', IQRTransformer()),       # Ausreißerbehandlung
    ('imputer', SimpleImputer('median')),# Fehlwert-Imputation
    ('scaler', StandardScaler())         # Standardisierung
])
```

### 3. Modellvergleich

| Modell | Recall (CV) | Accuracy (CV) |
|---|---|---|
| Random Forest (default) | 48 % | 99 % |
| Random Forest (GridSearch) | **82 %** | 97 % |
| **Gaussian Naive Bayes** | **90 %** | 88 % |

**Fazit:** Naive Bayes schlägt Random Forest bei Recall deutlich und wurde als finales Modell gewählt.

### 4. Hyperparameter-Tuning
- Random Forest: `GridSearchCV` → beste Parameter: `max_depth=10`, `class_weight=balanced`
- Naive Bayes: `RandomizedSearchCV` über `var_smoothing`

---

## 🗂️ Projektstruktur

```
📁 scania-predictive-maintenance/
├── issaoui_mL.ipynb              # Hauptnotebook (EDA + Modellierung)
├── scania_aps_model_gnb.pkl      # Gespeichertes Modell
├── scania_aps_pipeline.pkl       # Gespeicherte Pipeline
├── aps_failure_training_set.csv  # Trainingsdaten
└── README.md
```

---

## 🛠️ Tech Stack

- **Python** 3.x
- **scikit-learn** – Pipeline, GridSearchCV, GaussianNB, RandomForestClassifier
- **pandas / NumPy** – Datenverarbeitung
- **Matplotlib / Seaborn** – Visualisierung
- **joblib** – Modell-Serialisierung

---

## 📦 Verwendung

```python
import joblib
import pandas as pd

# Modell und Pipeline laden
pipeline = joblib.load("scania_aps_pipeline.pkl")
model = joblib.load("scania_aps_model_gnb.pkl")

# Vorhersage
X_prepared = pipeline.transform(X_new)
predictions = model.predict(X_prepared)
# 0 = in Ordnung (neg), 1 = APS defekt (pos)
```

---

## 📚 Datenquelle

Scania CV AB — [APS Failure at Scania Trucks](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)  
IDA Industrial Challenge 2016

---

## 👤 Autor

**Atef Issaoui**  
Maschinenbau-Student (berufsbegleitend), Hochschule Rhein-Main  
ML-Weiterbildung, Technische Hochschule Deggendorf  
📧 issaouiatef@gmail.com
