# Evaluation Report — Sensor Anomaly Detection

## Run Info
- **Date:** 2026-05-09 01:09
- **Commit:** unknown
- **Params:** window_size=128, lr=0.001, epochs=50

## Classification Report

```
                  precision    recall  f1-score   support

          Normal       1.00      1.00      1.00       952
Inner Race Fault       0.86      0.98      0.92       941
Outer Race Fault       0.91      0.83      0.86       900
      Ball Fault       0.92      0.86      0.89       927

        accuracy                           0.92      3720
       macro avg       0.92      0.92      0.92      3720
    weighted avg       0.92      0.92      0.92      3720

```

## Confusion Matrix

| | Normal | Inner Race | Outer Race | Ball Fault |
|---|---|---|---|---|
| Normal | 951 | 1 | 0 | 0 |
| Inner Race Fault | 3 | 926 | 9 | 3 |
| Outer Race Fault | 0 | 91 | 744 | 65 |
| Ball Fault | 0 | 59 | 69 | 799 |

## Reproducibility
- Run `dvc repro` to reproduce these exact results.
- DVC lock file: `dvc.lock` (committed)
- MLflow experiment: `sensor_anomaly_detection`
