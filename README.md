# Leakage-Resistant Early Warning for Lithium-Ion Battery Thermal Runaway

This project reproduces and stress-tests an XGBoost approach for identifying mechanically induced lithium-ion battery failure. Its main focus is evaluation rigor: heterogeneous sensor files are normalized, each battery test is kept within a single data split, and predictions are evaluated only before the first critical threshold crossing.

## Research Question

Can voltage, temperature, and mechanical force signals identify whether a critical battery condition will occur within the next 10 seconds?

A critical condition is operationally defined as either:

- cell voltage below 3.0 V; or
- measured temperature above 80 °C.

## Data

The source collection contains 210 mechanical-indentation battery experiments from Oak Ridge National Laboratory and Sandia National Laboratories. The workbooks include voltage, temperature, force or penetration, time, chemistry, capacity, and state-of-charge information.

Download the data from the [Mendeley Data repository](https://data.mendeley.com/datasets/sn2kv34r4h/1). The dataset is licensed under CC BY 4.0 and is not redistributed here.

The project was inspired by the 2026 paper [XGBoost-Powered Predictive Analytics for Early Identification of Thermal Runaway in Lithium-Ion Batteries](https://doi.org/10.3390/wevj17020068). This repository uses a stricter pre-onset warning formulation, so its results are not directly comparable with the paper's headline F1 score.

## Data Engineering

The raw workbooks do not share one universal schema. The pipeline:

1. scans every sheet rather than assuming the first sheet is valid;
2. detects and maps the two principal laboratory schemas;
3. converts sensor fields to numeric values;
4. resamples each experiment to one-second intervals;
5. interpolates isolated gaps within each battery test;
6. engineers voltage and temperature rates of change and interaction features;
7. excludes observations at or after the first critical threshold crossing.

Of the 210 workbooks, 209 contain a usable sensor sheet. Under the strict pre-onset definition, 207 tests contribute modeling observations.

## Leakage-Resistant Evaluation

The split is performed by battery-test file, not by individual sensor row:

- 70% training files;
- 15% validation files;
- 15% test files.

The probability threshold is selected using the validation set and then applied once to the held-out test files. Because the positive class represents only about 0.8% of pre-onset timestamps, F1 and precision-recall AUC are emphasized over accuracy.

## Preliminary Results

| Metric | Test result |
|---|---:|
| Precision | 0.425 |
| Recall | 0.423 |
| F1 | 0.424 |
| PR-AUC | 0.395 |
| ROC-AUC | 0.928 |

The strongest model features were temperature rate × voltage, temperature rate of change, voltage rate of change, voltage × force, and absolute temperature. The gap between ROC-AUC and PR-AUC illustrates why rare-event models should not be judged by accuracy or ROC-AUC alone.

## Repository Structure

```text
battery-thermal-runaway-early-warning/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
└── src/
    └── train_early_warning.py
```

## Run the Project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_early_warning.py --data-dir data/raw/excel
```

## Limitations and Next Steps

- The experiments intentionally induce mechanical failure and do not represent the true field prevalence of battery incidents.
- Mechanical force measurements are generally unavailable in a standard battery-management system.
- Performance should be evaluated by chemistry, laboratory, state of charge, and battery format.
- Add grouped cross-validation and confidence intervals.
- Compare 5-, 10-, and 20-second warning horizons.
- Test a voltage-and-temperature-only deployment model.

## Skills Demonstrated

Python, multi-schema Excel ingestion, time-series feature engineering, rare-event classification, XGBoost, file-level validation, leakage prevention, threshold selection, model interpretation, and scientific communication.
