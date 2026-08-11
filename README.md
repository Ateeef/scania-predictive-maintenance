# Predictive Maintenance – Scania APS Failure Detection

**Prüfungsarbeit im Rahmen des Hochschulzertifikats »Maschinelles Lernen«**  
TH Deggendorf · Bearbeitungszeit: 1 Monat

---

## Projektbeschreibung

Das Air Pressure System (APS) erzeugt Druckluft für Bremsen und Getriebe schwerer Scania-LKWs. Ein Ausfall führt direkt zum Stillstand des Fahrzeugs.

Ziel: Binäre Klassifikation auf Basis anonymisierter Sensordaten — APS-Defekt (pos) oder kein Defekt (neg) — mit Fokus auf kostenoptimale Fehlererkennung.

**Datensatz:** 60.000 Betriebsdatensätze · 170 anonymisierte Sensoren · IDA Industrial Challenge 2016

---

## Klassenverteilung

![Klassenverteilung](images/klassenverteilung.png)

Starkes Klassenungleichgewicht: Nur 1,7 % der Datensätze sind positiv (defekt). Fehlklassifikationen haben dabei sehr unterschiedliche Kosten — ein übersehener Defekt (Cost_2 = 500 €) ist 50× teurer als ein Fehlalarm (Cost_1 = 10 €). Die Metrik der Wahl ist deshalb **Recall**, nicht Accuracy.

---

## Datenaufbereitung

![Fehlende Werte](images/fehlende_werte.png)

170 Sensoren, viele davon nicht verwendbar. Nach systematischer Bereinigung:

- 28 Sensoren mit > 10 % fehlenden Werten entfernt
- 35 stark korrelierte (redundante) Sensoren entfernt
- 6 nahezu konstante Sensoren entfernt
- 1 vollständig konstanter Sensor entfernt

Ergebnis: **100 relevante Features** für das Modell.

---

## Explorative Datenanalyse

![Pairplot](images/pairplot.png)

Trotz Anonymisierung zeigen mehrere Sensoren klare Trennbarkeit zwischen den Klassen.

![Scatterplot](images/scatterplot.png)

Sensoren wie `aa_000` und `bx_000` zeigen strukturierte Unterschiede zwischen defekten und intakten Fahrzeugen.

![Boxplot](images/boxplot.png)

Defekte LKWs weisen in mehreren Sensoren systematisch andere Werteverteilungen auf — Basis für die Merkmalsauswahl.

---

## Modellvergleich

![Modellvergleich](images/modellvergleich.png)

| Modell | Recall | Accuracy |
|---|---|---|
| Random Forest | 58 % | 99 % |
| **Gaussian Naive Bayes** | **91 %** | **87 %** |

Der Random Forest optimiert auf Accuracy und übersieht dabei mehr als 40 % der Defekte. Gaussian Naive Bayes priorisiert Recall — die entscheidende Metrik für diesen Anwendungsfall.

---

## Ergebnis

![Confusion Matrix](images/confusion_matrix_final.png)

Testdaten: 12.000 Datensätze (ungesehen)

| | Vorhergesagt: neg | Vorhergesagt: pos |
|---|---|---|
| **Tatsächlich: neg** | 10.309 | 1.491 |
| **Tatsächlich: pos** | 19 | 181 |

**Recall: 91 %** — 181 von 200 Defekten erkannt.  
Geschätzte Kosteneinsparung gegenüber Random Forest: erheblich, da 76 zusätzliche Defekte korrekt identifiziert werden.

---

## Projektstruktur

```
scania-predictive-maintenance/
├── issaoui_mL.ipynb          # vollständige Analyse mit allen Plots
├── issaoui_mL.pdf            # Notebook als PDF
├── scania_praesentation_final.pptx  # Projektpräsentation
├── scania_aps_model_gnb.pkl  # trainiertes Modell
├── scania_aps_pipeline.pkl   # Preprocessing-Pipeline
├── images/                   # Plots aus der Analyse
└── README.md
```

---

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `joblib`

---

## Schnellstart

```python
import joblib

pipeline = joblib.load("scania_aps_pipeline.pkl")
model    = joblib.load("scania_aps_model_gnb.pkl")

X_prepared  = pipeline.transform(X_new)
predictions = model.predict(X_prepared)
# 0 = neg (kein Defekt)  |  1 = pos (APS defekt)
```

---

## Datensatz

APS Failure at Scania Trucks — Scania CV AB / UCI Machine Learning Repository (2016)  
IDA Industrial Challenge 2016  
[UCI Repository](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)

---

## Autor

**Atef Issaoui**  
Metallograph & Werkstoffanalytiker · Maschinenbau-Student (HS Rhein-Main)  
Zertifizierter Datenanalyst (Python)  
issaouiatef@gmail.com · [github.com/ateeef](https://github.com/ateeef)
