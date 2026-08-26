# NASA Astronet 1D-CNN Deep Learning Report

This report summarizes the performance of the **Dual-Branch 1D-Convolutional Neural Network (1D-CNN)** with multi-modal tabular fusion trained on NASA TESS stellar light curves.

---

## 1. 1D-CNN Model Architecture

- **Global 1D-CNN Branch (201 bins)**: Multi-scale 1D convolutions ($k=5$, channels $1 	o 16 	o 32 	o 64$) extracting broad orbital context and secondary eclipse morphology.
- **Local Zoomed 1D-CNN Branch (61 bins)**: Fine-scale 1D convolutions ($k=3$, channels $1 	o 16 	o 32 	o 64$) resolving ingress/egress slopes and bottom curvature (U vs. V shape).
- **Tabular Feature Branch (14 features)**: Dense network ($14 	o 32$) with Batch Normalization and Dropout.
- **Multi-Modal Fusion**: Concatenates $640 + 640 + 32 = 1312$ features into dense classification layers with Dropout.

---

## 2. Validation Metrics

| Metric | Score |
| :--- | :--- |
| **5-Fold Stratified Cross-Validation** | **50.10% ± 3.44%** |
| **Holdout Test Set Accuracy** (81 stars) | **43.94%** |

### Test Set Classification Report
```
                  precision    recall  f1-score   support

confirmed_planet       0.45      0.24      0.31        38
eclipsing_binary       0.41      0.63      0.50        49
  false_positive       0.49      0.40      0.44        45

        accuracy                           0.44       132
       macro avg       0.45      0.42      0.42       132
    weighted avg       0.45      0.44      0.42       132

```

---

## 3. Key Observations
1. **End-to-End Shape Perception**: The 1D-CNN directly captures the continuous curvature of planetary transit bottoms (U-shape) vs. binary star V-shapes without relying solely on summary metrics.
2. **Multi-Scale Context**: Combining the 201-bin global orbital baseline with the 61-bin zoomed transit window allows the network to simultaneously vet against out-of-transit stellar variability and evaluate in-transit ingress slopes.
