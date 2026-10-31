# NASA Astronet 1D-CNN Deep Learning Report

This report summarizes the performance of the **Dual-Branch 1D-Convolutional Neural Network (1D-CNN)** with multi-modal tabular fusion evaluated using 5-fold Stratified Cross-Validation out-of-fold predictions across all **1393 stars** with strict zero data leakage.

---

## 1. 1D-CNN Model Architecture

* **Global 1D-CNN Branch (201 bins)**: Multi-scale 1D convolutions ($k=5$, channels $1 \to 16 \to 32 \to 64$) extracting broad orbital context and secondary eclipse morphology.
* **Local Zoomed 1D-CNN Branch (61 bins)**: Fine-scale 1D convolutions ($k=3$, channels $1 \to 16 \to 32 \to 64$) resolving ingress and egress slopes and transit curvature.
* **Tabular Feature Branch (22 features)**: Dense network (22 $\to 32$) with Batch Normalization and Dropout.
* **Multi-Modal Fusion**: Concatenates $640 + 640 + 32 = 1312$ latent features into dense classification layers with Dropout.

---

## 2. Validation Metrics (5-Fold Stratified Cross-Validation)

| Metric | Score |
| :--- | :--- |
| **5-Fold Cross-Validation Accuracy** | **52.62% ± 2.63%** |
| **Overall Out-Of-Fold Accuracy** (1393 stars) | **52.62%** |

### Out-Of-Fold Classification Report across All 1393 Stars
```
                  precision    recall  f1-score   support

confirmed_planet       0.58      0.64      0.61       469
eclipsing_binary       0.50      0.62      0.56       506
  false_positive       0.47      0.29      0.36       418

        accuracy                           0.53      1393
       macro avg       0.52      0.52      0.51      1393
    weighted avg       0.52      0.53      0.51      1393

```

---

## 3. Data Integrity & Leakage Prevention
1. **Zero Data Leakage**: Standard scaling and class balancing weights were fitted strictly on the training partition of each fold and applied to held-out validation stars.
2. **True Out-Of-Fold Output**: Every prediction stored in `outputs/predictions.csv` was generated when the star was in the held-out validation partition.
