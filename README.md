# Analyse aktueller Methoden zur Prognose von Day-Ahead-Strompreisen

**Ein empirischer Vergleich statistischer, Machine-Learning- und Deep-Learning-Modelle**

Projektarbeit im Studiengang **Wirtschaftsingenieurwesen**  
Friedrich-Alexander-Universität Erlangen-Nürnberg  
Lehrstuhl für Fertigungsautomatisierung und Produktionssystematik 

Von Manuel Hilgers

---

## Über dieses Projekt

Diese Arbeit entwickelt ein modulares, reproduzierbares Vergleichssystem für vier methodisch
unterschiedliche Modellklassen zur Day-Ahead-Strompreisprognose. Als Datengrundlage dient
eine eigenständig aufgebaute ENTSO-E-Datenbasis für die Märkte Deutschland/Luxemburg,
Frankreich und Spanien im Zeitraum 2020 bis 2025. Alle Modelle werden unter einheitlicher
Trainings- und Testlogik auf einem gemeinsamen Out-of-Sample-Zeitraum (11.01.2024–31.12.2025)
evaluiert.

---

## Modelle

| Modell | Klasse | Beschreibung |
|---|---|---|
| **LEAR** | Statistisch | Lasso-Estimated AutoRegressive Model, rollendes 4-Wochen-Fenster |
| **DNN** | Deep Learning | Deep Neural Network, 2 Hidden Layers, Hyperparameter-Optimierung via Hyperband |
| **RF** | Machine Learning | Random Forest mit rollendem Trainingsfenster |
| **SVR** | Machine Learning | Support Vector Regression mit rollendem Trainingsfenster |
| **Naive** | Benchmark | Saisonaler Naive-Benchmark (Preise von d−7) |

---

## Märkte & Datenbasis

| Markt | Kürzel | Zeitraum gesamt | Out-of-Sample-Testperiode |
|---|---|---|---|
| Deutschland / Luxemburg | DE-LU | 2020–2025 | 11.01.2024–31.12.2025 |
| Frankreich | FR | 2020–2025 | 11.01.2024–31.12.2025 |
| Spanien | ES | 2020–2025 | 11.01.2024–31.12.2025 |

Datenquelle: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)

---

## Projektstruktur

**Datenvorbereitung**

    00_entsoe_download_and_prepare.py     Datendownload via ENTSO-E API
    01_prepare_clean_from_raw_only.py     Datenbereinigung (ohne API-Zugang)
    02_violin_plots.py                    EDA – Violinplots Preisverteilung
    03_preiszeitreihen_plot.py            EDA – Zeitreihenplots Rohpreise
    make_train_val_test_splits.py         Train/Validation/Test-Splits erzeugen
    build_features_rf_svr.py             Feature Engineering für RF und SVR

**Modelltraining & Forecasting**

    run_lear.py                           LEAR: Training und Prognose
    run_dnn_hyperopt.py                   DNN: Hyperparameter-Optimierung
    run_dnn_evaluation_all_markets.py     DNN: Evaluation auf allen Märkten
    run_rf_rolling_evaluation.py          RF: Rolling-Window-Evaluation
    run_svr_rolling_evaluation.py         SVR: Rolling-Window-Evaluation

**Auswertung & Visualisierung**

    evaluate_model_comparison.py          Metriken-Vergleich aller Modelle
    run_dm_tests.py                       Diebold-Mariano-Signifikanztests
    create_main_text_plots.py             Abbildungen für den Haupttext
    create_appendix_full_series.py        Anhang: vollständige Zeitreihen
    create_distribution_barplots.py       Fehlerverteilungs-Barplots

---

## Installation & Umgebung

    conda env create -f epf_environment.yml
    conda activate epf

Zentrale Abhängigkeiten: `epftoolbox`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `hyperopt`, `tensorflow`

---

## Ausführungsreihenfolge

    1.  00_entsoe_download_and_prepare.py        Rohdaten laden (ENTSO-E API-Key erforderlich)
        alternativ: 01_prepare_clean_from_raw_only.py
    2.  make_train_val_test_splits.py            Datensplits anlegen
    3.  build_features_rf_svr.py                Feature Engineering für RF/SVR
    4.  run_lear.py                              LEAR prognostizieren
    5.  run_dnn_hyperopt.py                      DNN-Hyperparameter optimieren
    6.  run_dnn_evaluation_all_markets.py        DNN evaluieren
    7.  run_rf_rolling_evaluation.py             RF evaluieren
    8.  run_svr_rolling_evaluation.py            SVR evaluieren
    9.  evaluate_model_comparison.py             Alle Metriken zusammenführen
    10. run_dm_tests.py                          Statistische Signifikanztests
    11. create_main_text_plots.py                Plots generieren
        create_appendix_full_series.py
        create_distribution_barplots.py

---

## Evaluationsmetriken

- **MAE** – Mean Absolute Error
- **rMAE** – Relative MAE (normiert auf Naive-Benchmark, nach Lago et al.)
- **SMAPE** – Symmetric Mean Absolute Percentage Error
- **MAPE** – Mean Absolute Percentage Error
- **Diebold-Mariano-Test** – statistische Signifikanzprüfung der Prognosegüteunterschiede
