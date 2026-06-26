# Validation Metrics Report (ExoFinder Classifier Tuning)

This report was automatically generated on the final execution of the training pipeline. It compares the performance metrics of four tuned classification models: **Random Forest**, **XGBoost**, **Logistic Regression** (baseline), and **Ensemble (Soft Voting)**.

---

## 1. Classifier Performance Comparison

Below is the comparison of performance metrics of the models on the 12 tabular features:

| Metric | Random Forest | XGBoost | Logistic Regression | Ensemble |
| :--- | :--- | :--- | :--- | :--- |
| **Train Accuracy** (324 stars) | 71.9% | 97.2% | 48.8% | 90.1% |
| **Test Accuracy** (81 stars) | 40.7% | 46.9% | 48.1% | 49.4% |
| **5-Fold Stratified CV Accuracy** | **44.0% ± 2.2%** | **45.4% ± 3.9%** | **41.2% ± 7.8%** | **44.9% ± 4.0%** |

### Selected Best Model for Deployment: **XGBoost**
The model with the highest 5-Fold CV accuracy (**45.4%**) was selected and saved to `outputs/model.pkl`.

---

## 2. Test Set Classification Reports

### Random Forest
```
                  precision    recall  f1-score   support

confirmed_planet       0.52      0.56      0.54        27
eclipsing_binary       0.33      0.30      0.31        27
  false_positive       0.36      0.37      0.36        27

        accuracy                           0.41        81
       macro avg       0.40      0.41      0.40        81
    weighted avg       0.40      0.41      0.40        81

```

### XGBoost
```
                  precision    recall  f1-score   support

confirmed_planet       0.50      0.63      0.56        27
eclipsing_binary       0.43      0.22      0.29        27
  false_positive       0.45      0.56      0.50        27

        accuracy                           0.47        81
       macro avg       0.46      0.47      0.45        81
    weighted avg       0.46      0.47      0.45        81

```

### Logistic Regression
```
                  precision    recall  f1-score   support

confirmed_planet       0.40      0.44      0.42        27
eclipsing_binary       0.54      0.48      0.51        27
  false_positive       0.52      0.52      0.52        27

        accuracy                           0.48        81
       macro avg       0.49      0.48      0.48        81
    weighted avg       0.49      0.48      0.48        81

```

### Ensemble (Soft Voting)
```
                  precision    recall  f1-score   support

confirmed_planet       0.52      0.59      0.55        27
eclipsing_binary       0.47      0.33      0.39        27
  false_positive       0.48      0.56      0.52        27

        accuracy                           0.49        81
       macro avg       0.49      0.49      0.49        81
    weighted avg       0.49      0.49      0.49        81

```

---

## 3. Discussion & Observations
- **Regularization & Scaling**: Standard scaling was applied to all features, fitting only on the training set/fold. Strict regularization parameters were added to both Random Forest (`max_depth=4`, `min_samples_leaf=3`) and XGBoost (`subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.3`, `reg_lambda=1.5`), which drastically reduces overfitting compared to previous runs.
- **Class Balancing**: Balanced weights were incorporated for all models (including XGBoost sample weights). SMOTE oversampling was applied to the training set when `imbalanced-learn` is available, ensuring the classifier is not biased towards any dominant class.
- **Ensemble**: A soft-voting ensemble combines predictions from all three models, weighted by their individual CV accuracies.
- **Feature Set**: The model uses a 12-feature set focusing on transit physical shape and significance: bls_depth, bls_duration, bls_power, transit_snr, mean_to_max_depth_ratio, in_transit_skew, in_transit_kurtosis, flux_kurtosis, ingress_egress_symmetry, secondary_eclipse_check, period_ratio, depth_to_noise.
