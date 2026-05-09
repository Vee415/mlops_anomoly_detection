# Evaluation Report — Sensor Anomaly Detection

## Run Info
- **Date:** 2026-05-09 13:30
- **Commit:** 0a541fa
- **Architecture:** cnn
- **Params:** window_size=128, lr=0.001, epochs=50

## Classification Report

```
                  precision    recall  f1-score   support

          Normal       1.00      1.00      1.00       930
Inner Race Fault       1.00      1.00      1.00       930
Outer Race Fault       1.00      1.00      1.00       930
      Ball Fault       1.00      0.99      1.00       930

        accuracy                           1.00      3720
       macro avg       1.00      1.00      1.00      3720
    weighted avg       1.00      1.00      1.00      3720

```

## Confusion Matrix

| | Normal | Inner Race | Outer Race | Ball Fault |
|---|---|---|---|---|
| Normal | 930 | 0 | 0 | 0 |
| Inner Race Fault | 0 | 930 | 0 | 0 |
| Outer Race Fault | 0 | 0 | 927 | 3 |
| Ball Fault | 0 | 2 | 4 | 924 |

## Reproducibility
- Run `dvc repro` to reproduce these exact results.
- DVC lock file: `dvc.lock` (committed)
- MLflow experiment: `sensor_anomaly_detection`
