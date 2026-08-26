# Validation Metrics Report (ExoFinder Stacking & Calibration)

This report was automatically generated following hyperparameter optimization, stacking ensembling, and probability calibration across the 405-target dataset.

---

## 1. Classifier Performance Comparison

Below is the comparison of performance metrics of all tuned models across the **14 physical features**:

| Model Architecture | Train Accuracy (525 stars) | Test Accuracy (132 stars) | 5-Fold Stratified CV Accuracy |
| :--- | :--- | :--- | :--- |
| **Random Forest (Tuned)** | 82.9% | 50.8% | **46.7% ± 4.0%** |
| **XGBoost (Tuned)** | 81.7% | 47.7% | **45.9% ± 4.5%** |
| **Logistic Regression (Tuned)** | 44.0% | 34.8% | **41.9% ± 2.8%** |
| **Stacking Classifier (Meta-LR)** | 81.5% | 50.8% | **47.6% ± 5.2%** |
| **Calibrated Stacking (Platt Sigmoid)** | 80.6% | 50.0% | **46.9% ± 5.0%** |

### Selected Deployed Model: **Calibrated Stacking**
Saved to `outputs/model.pkl` with Platt-calibrated probability mappings.

---

## 2. Test Set Classification Reports

### Calibrated Stacking Ensemble (Production Model)
```
                  precision    recall  f1-score   support

confirmed_planet       0.52      0.37      0.43        38
eclipsing_binary       0.49      0.73      0.59        49
  false_positive       0.52      0.36      0.42        45

        accuracy                           0.50       132
       macro avg       0.51      0.49      0.48       132
    weighted avg       0.51      0.50      0.48       132
```

### Stacking Ensemble (Pre-calibration)
```
                  precision    recall  f1-score   support

confirmed_planet       0.54      0.37      0.44        38
eclipsing_binary       0.50      0.69      0.58        49
  false_positive       0.50      0.42      0.46        45

        accuracy                           0.51       132
       macro avg       0.51      0.49      0.49       132
    weighted avg       0.51      0.51      0.50       132
```

### Tuned XGBoost
```
                  precision    recall  f1-score   support

confirmed_planet       0.53      0.50      0.51        38
eclipsing_binary       0.47      0.55      0.51        49
  false_positive       0.44      0.38      0.40        45

        accuracy                           0.48       132
       macro avg       0.48      0.48      0.48       132
    weighted avg       0.48      0.48      0.47       132
```

### Tuned Random Forest
```
                  precision    recall  f1-score   support

confirmed_planet       0.50      0.39      0.44        38
eclipsing_binary       0.52      0.61      0.56        49
  false_positive       0.50      0.49      0.49        45

        accuracy                           0.51       132
       macro avg       0.51      0.50      0.50       132
    weighted avg       0.51      0.51      0.50       132
```

---

## 3. Key Enhancements & Methodology
1. **Odd/Even Transit Depth Analysis**: Evaluates odd-cycle vs. even-cycle transit depth ratios and difference significance (sigma) to separate Eclipsing Binaries from transiting planets.
2. **Automated Hyperparameter Optimization**: `RandomizedSearchCV` explores regularization, tree depth, and learning rates under stratified cross-validation.
3. **Stacking Ensemble**: Replaced manual heuristic voting with `StackingClassifier`, allowing a Logistic Regression meta-learner to learn optimal non-linear blending weights.
4. **Probability Calibration**: `CalibratedClassifierCV` transforms raw model decision margins into well-calibrated posterior probabilities.
5. **Physical Feature Set (14 features)**: bls_depth, bls_duration, bls_power, transit_snr, mean_to_max_depth_ratio, in_transit_skew, in_transit_kurtosis, flux_kurtosis, ingress_egress_symmetry, secondary_eclipse_check, period_ratio, depth_to_noise, odd_even_depth_ratio, odd_even_mismatch_sigma.
