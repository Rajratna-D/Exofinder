# Validation Metrics Report (ExoFinder Stacking & Calibration)

This report was generated using 5-fold Stratified Cross-Validation out-of-fold predictions across all **1393 verified NASA TESS stars** with strict zero data leakage.

---

## 1. Classifier Performance Comparison

Below is the comparison of out-of-fold metrics across all **22 physical features**:

| Model Architecture | 5-Fold Stratified CV Accuracy | 5-Fold Stratified Macro F1 |
| :--- | :--- | :--- |
| **Random Forest (Regularized)** | **50.18% ± 2.14%** | 47.7% |
| **XGBoost (Regularized)** | **51.97% ± 2.63%** | 48.3% |
| **Calibrated Stacking (Production)** | **51.76% ± 2.87%** | **48.51%** |

### Selected Deployed Model: **Calibrated Stacking**
Saved to `outputs/model.pkl` with Platt-calibrated probability mappings.

---

## 2. Out-Of-Fold Classification Report across All 1393 Stars

```
                  precision    recall  f1-score   support

confirmed_planet       0.54      0.67      0.60       469
eclipsing_binary       0.49      0.63      0.55       506
  false_positive       0.53      0.21      0.30       418

        accuracy                           0.52      1393
       macro avg       0.52      0.50      0.49      1393
    weighted avg       0.52      0.52      0.49      1393

```

---

## 3. Data Integrity & Anti-Leakage Verification
1. **Zero Data Leakage**: Standard scaling and label transformations are fitted strictly within each training fold and applied to held-out validation stars.
2. **True Out-Of-Fold Persistence**: The `outputs/predictions.csv` matrix contains 100% out-of-fold predictions, ensuring the deployed catalog reflects genuine generalization.
3. **Overfitting Mitigation**: Tree depths constrained to 4 with minimum leaf count of 5 and $L_2$ regularization $\lambda = 3.0$.
