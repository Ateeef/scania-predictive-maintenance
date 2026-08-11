# Predictive Maintenance – Scania APS Failure Detection

**Prüfungsarbeit im Rahmen des Hochschulzertifikats »Maschinelles Lernen«**  
TH Deggendorf · Bearbeitungszeit: 1 Monat

---

## Das Problem

Das Air Pressure System (APS) erzeugt Druckluft für Bremsen und Getriebe schwerer Scania-LKWs. Fällt es aus, steht der LKW — mitten auf der Autobahn.

Die Frage: Lässt sich ein APS-Defekt anhand von Sensordaten vorhersagen, bevor er passiert?

Dabei gibt es eine entscheidende Einschränkung: Nicht jeder Fehler wiegt gleich schwer.

- Ein Fehlalarm (intakter LKW wird zur Werkstatt geschickt): **10 €**
- Ein übersehener Defekt (defekter LKW fährt weiter): **500 €**

Das bedeutet: klassische Accuracy ist hier die falsche Metrik. Ein Modell das 98 % Accuracy hat aber jeden zweiten Defekt übersieht, ist im echten Betrieb wertlos. Gesucht wird **Recall**.

---

## Die Daten — und warum sie schwierig sind

Der Datensatz umfasst 60.000 Betriebsdatensätze mit 170 Sensoren. Erster Blick auf die Zielklasse:

![Klassenverteilung](images/klassenverteilung.png)

Nur 1,7 % der Datensätze zeigen einen echten Defekt. Das Modell lernt auf 98,3 % intakten LKWs — und soll trotzdem zuverlässig die 1,7 % finden. Klassisches Ungleichgewichtsproblem.

Dazu kommt eine zweite Herausforderung: Alle 170 Sensoren sind **anonymisiert**. Kein Sensor trägt einen Namen, der auf seine physikalische Bedeutung hinweist — nur Codes wie `aa_000`, `ci_000`, `bx_000`. Die Analyse musste rein datenbasiert laufen.

---

## Datenbereinigung — von 170 auf 100 Sensoren

Bevor überhaupt ein Muster gesucht werden konnte, musste der Datensatz bereinigt werden:

![Fehlende Werte](images/fehlende_werte.png)

Viele Sensoren lieferten kaum verwertbare Daten. Nach systematischer Prüfung wurden entfernt:

- 28 Sensoren mit mehr als 10 % fehlenden Werten
- 35 stark korrelierte (redundante) Sensoren
- 6 nahezu konstante Sensoren
- 1 vollständig konstanter Sensor (`cd_000`)

Übrig blieben **100 Sensoren** — die Grundlage für die weitere Analyse.

---

## Mustererkennung — gibt es überhaupt trennbare Signale?

Erst nach der Bereinigung konnte die eigentliche Frage gestellt werden: Zeigen defekte LKWs in den Daten ein anderes Muster als intakte?

![Pairplot](images/pairplot.png)

Ja — trotz Anonymisierung. Im Pairplot der fünf stärksten Sensoren (u.a. `ci_000`, `bb_000`, `bv_000`) bilden defekte LKWs (rot) und intakte (grün) in mehreren Sensorpaaren klar trennbare Cluster.

Das war der entscheidende Befund: Die Signale sind vorhanden, sie müssen nur richtig ausgewertet werden.

![Scatterplot](images/scatterplot.png)

Besonders `aa_000` und `bx_000` zeigen eine ausgeprägte Trennung — defekte LKWs liegen systematisch in anderen Wertebereichen als intakte.

![Boxplot](images/boxplot.png)

Der Boxplot bestätigt es: Die Werteverteilungen unterscheiden sich strukturell. Kein Zufall — das Modell hat echte physikalische Signale gefunden, auch ohne zu wissen was die Sensoren messen.

---

## Modellauswahl — Accuracy ist nicht alles

Zwei Modelle wurden trainiert und direkt verglichen:

![Modellvergleich](images/modellvergleich.png)

| Modell | Recall | Accuracy |
|---|---|---|
| Random Forest | 58 % | 99 % |
| **Gaussian Naive Bayes** | **91 %** | **87 %** |

Der Random Forest sieht auf den ersten Blick besser aus — 99 % Accuracy. Aber er übersieht mehr als 40 % aller Defekte. Bei 500 € pro übersehenen Fall ist das im echten Betrieb nicht akzeptabel.

Gaussian Naive Bayes wurde auf **Recall** optimiert und findet 91 % der Defekte. Dafür nimmt es mehr Fehlalarme in Kauf — die aber nur 10 € kosten. Das war die richtige Abwägung.

---

## Ergebnis

![Confusion Matrix](images/confusion_matrix_final.png)

Getestet auf 12.000 ungesehenen Datensätzen:

| | Vorhergesagt: neg | Vorhergesagt: pos |
|---|---|---|
| **Tatsächlich: neg** | 10.309 | 1.491 |
| **Tatsächlich: pos** | 19 | 181 |

**181 von 200 Defekten erkannt. Recall: 91 %.**

19 Defekte wurden übersehen (9.500 €). 1.491 Fehlalarme wurden ausgelöst (14.910 €). Gesamtkosten: ~24.410 €.

Zum Vergleich: Der Random Forest hätte bei 58 % Recall rund 84 Defekte übersehen — das entspricht 42.000 € allein an übersehenen Schäden.

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
