# Predictive Maintenance – Scania APS Failure Detection

**Prüfungsarbeit im Rahmen des Hochschulzertifikats »Maschinelles Lernen«**  
TH Deggendorf · Bearbeitungszeit: 1 Monat

---

## Was steckt hinter diesem Projekt?

Das Air Pressure System (APS) ist ein sicherheitskritisches System in schweren Scania-LKWs — es erzeugt Druckluft für Bremsen und Getriebe. Fällt es aus, steht der LKW.

Die Aufgabe war klar: Kann man anhand von Sensordaten vorhersagen, ob ein LKW ein APS-Problem hat — **bevor** es zum Ausfall kommt?

Ich habe 60.000 echte Betriebsdatensätze analysiert, ein Machine-Learning-Modell entwickelt und am Ende **91 % der defekten LKWs erkannt**.

---

## Die Herausforderung: Nadel im Heuhaufen

Das erste was ich beim Öffnen der Daten gesehen habe — ein extremes Ungleichgewicht:

![Klassenverteilung](images/klassenverteilung.png)

Von 60.000 LKWs sind nur **1.000 wirklich defekt** (1,7 %). Das Modell muss genau diese 1,7 % finden, ohne bei den anderen ständig Fehlalarm zu schlagen.

Dazu kam eine weitere Herausforderung: Alle 170 Sensoren sind **anonymisiert**. Keine Bezeichnungen, keine physikalische Bedeutung — nur Codes wie `aa_000`, `bb_000` usw. Die komplette Analyse musste rein datenbasiert laufen.

---

## Was ich gemacht habe

### Schritt 1 — Daten verstehen und bereinigen

170 Sensorsignale, viele davon unbrauchbar:

![Fehlende Werte](images/fehlende_werte.png)

Manche Sensoren hatten über 80 % fehlende Werte. Andere lieferten konstante Werte oder doppelte Informationen. Nach der Bereinigung blieben **100 relevante Sensoren** übrig.

Entfernt wurden:
- 28 Sensoren mit mehr als 10 % fehlenden Werten
- 35 stark korrelierte (redundante) Sensoren
- 6 nahezu konstante Sensoren
- 1 vollständig konstanter Sensor

### Schritt 2 — Muster erkennen

Trotz Anonymisierung zeigen bestimmte Sensoren bei defektem APS klare Auffälligkeiten. Die stärksten Signale trennen defekte von intakten LKWs sauber — ohne dass ich weiß, was der Sensor physikalisch misst.

![Pairplot](images/pairplot.png)

Im Pairplot sieht man deutlich: Defekte LKWs (rot) und intakte (grün) bilden in bestimmten Sensorpaaren klar trennbare Cluster. Das war die Grundlage für die Merkmalsauswahl.

![Scatterplot](images/scatterplot.png)

Einzelne Sensoren wie `aa_000` und `bx_000` zeigen eine ausgeprägte Trennung zwischen den Klassen — trotz fehlender physikalischer Beschriftung.

![Boxplot](images/boxplot.png)

Der Boxplot macht es noch klarer: Bei defekten LKWs liegen die Messwerte systematisch in anderen Bereichen. Kein Zufall — das Modell hat echte Signale gefunden.

### Schritt 3 — Modell entwickeln und vergleichen

Ich habe zwei Modelle gebaut und direkt verglichen:

![Modellvergleich](images/modellvergleich.png)

| Modell | Recall (Defekte erkannt) | Accuracy |
|---|---|---|
| Random Forest | 58 % | 99 % |
| **Gaussian Naive Bayes** | **91 %** | **87 %** |

Der Random Forest hatte zwar eine höhere Accuracy — aber er hat mehr als die Hälfte der defekten LKWs **übersehen**. Das ist in diesem Kontext gefährlich und teuer.

Naive Bayes hat 91 % der Defekte gefunden. Das war die Entscheidung.

---

## Das Ergebnis

![Finale Confusion Matrix](images/confusion_matrix_final.png)

Auf **12.000 unbekannten Testdaten**:

- **181 von 200** defekten LKWs korrekt erkannt
- 19 Defekte übersehen — das sind die teuren Fälle (je ~500 €)
- 1.491 Fehlalarme — ärgerlich, aber handhabbar (je ~10 €)

Ein übersehener Defekt kostet **50× mehr** als ein Fehlalarm. Das Modell ist auf diesen Trade-off ausgelegt.

---

## Warum das für Ingenieure relevant ist

> Geplante Wartung statt Panneneinsatz auf der Autobahn.

Predictive Maintenance bedeutet: Der LKW kommt in die Werkstatt, weil ein Algorithmus es sagt — nicht weil er liegengeblieben ist. Das spart Kosten, Zeit und im schlimmsten Fall Menschenleben.

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

X_prepared   = pipeline.transform(X_new)
predictions  = model.predict(X_prepared)
# 0 = APS in Ordnung  |  1 = APS defekt
```

---

## Datensatz

APS Failure at Scania Trucks — Scania CV AB / UCI Machine Learning Repository (2016)  
IDA Industrial Challenge 2016

---

## Autor

**Atef Issaoui**  
Metallograph & Werkstoffanalytiker · Maschinenbau-Student (HS Rhein-Main)  
Zertifizierter Datenanalyst (Python)  
issaouiatef@gmail.com · [github.com/ateeef](https://github.com/ateeef)
