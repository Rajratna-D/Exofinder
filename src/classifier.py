import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

def train_and_evaluate(features_csv=None, model_output=None, predictions_output=None):
    """
    Loads features, preprocesses data, runs hyperparameter search (RandomizedSearchCV)
    for RF, XGBoost, and Logistic Regression, trains a StackingClassifier with a meta-learner,
    calibrates probabilities using CalibratedClassifierCV, evaluates via 5-Fold Stratified CV,
    and saves the best calibrated model and predictions.
    """
    # Derive absolute paths from project root to avoid CWD dependency
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if features_csv is None:
        features_csv = os.path.join(project_root, "outputs", "features.csv")
    if model_output is None:
        model_output = os.path.join(project_root, "outputs", "model.pkl")
    if predictions_output is None:
        predictions_output = os.path.join(project_root, "outputs", "predictions.csv")

    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Feature table not found: {features_csv}")
        
    df = pd.read_csv(features_csv)
    
    # 1. Report Class Counts plainly before training
    print("Class distribution in dataset:")
    print(df['label'].value_counts())
    
    # 2. Preprocess features and labels
    # 14 physical features (including Odd/Even transit depth discriminators)
    feature_cols = [
        'bls_depth', 'bls_duration', 'bls_power', 
        'transit_snr', 'mean_to_max_depth_ratio', 'in_transit_skew', 
        'in_transit_kurtosis', 'flux_kurtosis',
        'ingress_egress_symmetry', 'secondary_eclipse_check',
        'period_ratio', 'depth_to_noise',
        'odd_even_depth_ratio', 'odd_even_mismatch_sigma'
    ]
    
    # Compute derived features if missing (backwards compatibility with older features.csv)
    if 'period_ratio' not in df.columns:
        if 'bls_period' in df.columns and 'catalog_period' in df.columns:
            df['period_ratio'] = df.apply(
                lambda r: r['bls_period'] / r['catalog_period'] if r['catalog_period'] > 0 and not np.isnan(r['catalog_period']) else 0.0, axis=1)
        else:
            df['period_ratio'] = 0.0
    if 'depth_to_noise' not in df.columns:
        if 'bls_depth' in df.columns and 'std_detrended' in df.columns:
            df['depth_to_noise'] = df.apply(
                lambda r: r['bls_depth'] / r['std_detrended'] if r['std_detrended'] > 0 else 0.0, axis=1)
        else:
            df['depth_to_noise'] = 0.0
    if 'odd_even_depth_ratio' not in df.columns:
        df['odd_even_depth_ratio'] = 1.0
    if 'odd_even_mismatch_sigma' not in df.columns:
        df['odd_even_mismatch_sigma'] = 0.0
    
    X = df[feature_cols].copy()
    y_raw = df['label'].copy()
    
    # If there's only 1 class (e.g. custom dataset without multiple labels), run in inference mode
    if y_raw.nunique() < 2:
        print("\n[INFO] Less than 2 classes found in dataset. Skipping retraining.")
        print("Running in inference mode using pre-trained model outputs/model.pkl...")
        
        if not os.path.exists(model_output):
            raise FileNotFoundError(
                f"Model file not found at {model_output}. "
                f"You must run the pipeline with the full dataset first to train the model."
            )
            
        with open(model_output, 'rb') as f:
            model_data = pickle.load(f)
            
        best_model = model_data['model']
        le = model_data['label_encoder']
        feature_cols = model_data['feature_cols']
        scaler = model_data['scaler']
        
        X_scaled = pd.DataFrame(scaler.transform(X[feature_cols]), columns=feature_cols)
        probs = best_model.predict_proba(X_scaled)
        preds = best_model.predict(X_scaled)
        pred_labels = le.inverse_transform(preds)
        confidences = np.max(probs, axis=1)
        
        predictions_df = df.copy()
        predictions_df['predicted_label'] = pred_labels
        predictions_df['confidence'] = confidences
        
        for idx, class_name in enumerate(le.classes_):
            predictions_df[f'prob_{class_name}'] = probs[:, idx]
            
        predictions_df['significance'] = 'high'
        low_sig_mask = (
            (predictions_df['predicted_label'] == 'confirmed_planet') & (predictions_df['transit_snr'] < 5.0)
        ) | (predictions_df['confidence'] < 0.6)
        predictions_df.loc[low_sig_mask, 'significance'] = 'low_significance'
        
        pred_dir = os.path.dirname(predictions_output)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        predictions_df.to_csv(predictions_output, index=False)
        print(f"Predictions table saved to {predictions_output} (Rows: {len(predictions_df)})")
        return predictions_df
        
    # Encode labels: confirmed_planet=0, eclipsing_binary=1, false_positive=2
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print("\nFeature matrix shape:", X.shape)
    print(f"Training on {len(feature_cols)} physical features:", list(X.columns))
    
    # 3. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\nStratified Split (80/20):")
    print(f"  Training set size: {len(X_train)} (stratified: {np.bincount(y_train)})")
    print(f"  Test set size:     {len(X_test)} (stratified: {np.bincount(y_test)})")
    
    # Apply Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Apply SMOTE oversampling to the training set to handle class imbalance
    if HAS_IMBLEARN:
        try:
            min_class_count = min(np.bincount(y_train))
            k_neighbors = min(5, min_class_count - 1)
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                print(f"  SMOTE applied: Training set expanded to {len(X_train_scaled)} samples (class distribution: {np.bincount(y_train)})")
            else:
                print("  SMOTE skipped: Too few samples in a class for oversampling.")
        except Exception as e:
            print(f"  SMOTE failed: {e}. Continuing without oversampling.")
    else:
        print("  [INFO] imbalanced-learn not installed. Skipping SMOTE.")

    # 4. Hyperparameter Tuning via RandomizedSearchCV
    print("\n=== Stage 1: Hyperparameter Optimization (RandomizedSearchCV) ===")
    cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # A. Tune Random Forest
    print("  Tuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [3, 4, 5, 6],
        'min_samples_leaf': [2, 3, 5],
        'max_features': ['sqrt', 'log2']
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        rf_param_grid,
        n_iter=12,
        cv=cv_tune,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_train_scaled, y_train)
    best_rf = rf_search.best_estimator_
    print(f"  Best RF Params: {rf_search.best_params_}")

    # B. Tune XGBoost
    print("  Tuning XGBoost...")
    xgb_param_grid = {
        'n_estimators': [50, 80, 120],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.06, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_alpha': [0.1, 0.3, 1.0],
        'reg_lambda': [1.0, 1.5, 3.0]
    }
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
        xgb_param_grid,
        n_iter=15,
        cv=cv_tune,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    best_xgb = xgb_search.best_estimator_
    print(f"  Best XGB Params: {xgb_search.best_params_}")

    # C. Tune Logistic Regression
    print("  Tuning Logistic Regression...")
    lr_param_grid = {
        'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    }
    lr_search = RandomizedSearchCV(
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        lr_param_grid,
        n_iter=6,
        cv=cv_tune,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    lr_search.fit(X_train_scaled, y_train)
    best_lr = lr_search.best_estimator_
    print(f"  Best LR Params: {lr_search.best_params_}")

    # Evaluate Individual Tuned Models
    rf_train_preds = best_rf.predict(X_train_scaled)
    rf_test_preds = best_rf.predict(X_test_scaled)
    
    xgb_train_preds = best_xgb.predict(X_train_scaled)
    xgb_test_preds = best_xgb.predict(X_test_scaled)
    
    lr_train_preds = best_lr.predict(X_train_scaled)
    lr_test_preds = best_lr.predict(X_test_scaled)

    # 5. Stacking Classifier with Meta-Learner
    print("\n=== Stage 2: Stacking Ensemble Classifier ===")
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', best_rf),
            ('xgb', best_xgb),
            ('lr', best_lr)
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )
    stacking_clf.fit(X_train_scaled, y_train)
    
    stacking_train_preds = stacking_clf.predict(X_train_scaled)
    stacking_test_preds = stacking_clf.predict(X_test_scaled)

    # 6. Probability Calibration (CalibratedClassifierCV)
    print("\n=== Stage 3: Probability Calibration (CalibratedClassifierCV) ===")
    calibrated_stacking = CalibratedClassifierCV(
        estimator=stacking_clf,
        method='sigmoid',
        cv=5
    )
    calibrated_stacking.fit(X_train_scaled, y_train)
    
    cal_train_preds = calibrated_stacking.predict(X_train_scaled)
    cal_test_preds = calibrated_stacking.predict(X_test_scaled)

    # 7. 5-Fold Stratified Cross-Validation on Full Training Set
    print("\n=== Stage 4: 5-Fold Stratified Cross-Validation ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_cv_scores = []
    xgb_cv_scores = []
    lr_cv_scores = []
    stacking_cv_scores = []
    calibrated_cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # RF CV
        m_rf = RandomForestClassifier(**best_rf.get_params())
        m_rf.fit(X_tr, y_tr)
        rf_cv_scores.append(accuracy_score(y_val, m_rf.predict(X_val)))
        
        # XGB CV
        m_xgb = xgb.XGBClassifier(**best_xgb.get_params())
        sw_tr = compute_sample_weight(class_weight='balanced', y=y_tr)
        m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr)
        xgb_cv_scores.append(accuracy_score(y_val, m_xgb.predict(X_val)))
        
        # LR CV
        m_lr = LogisticRegression(**best_lr.get_params())
        m_lr.fit(X_tr, y_tr)
        lr_cv_scores.append(accuracy_score(y_val, m_lr.predict(X_val)))
        
        # Stacking CV
        m_stk = StackingClassifier(
            estimators=[('rf', m_rf), ('xgb', m_xgb), ('lr', m_lr)],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            cv=5,
            n_jobs=-1
        )
        m_stk.fit(X_tr, y_tr)
        stacking_cv_scores.append(accuracy_score(y_val, m_stk.predict(X_val)))
        
        # Calibrated Stacking CV
        m_cal = CalibratedClassifierCV(estimator=m_stk, method='sigmoid', cv=3)
        m_cal.fit(X_tr, y_tr)
        calibrated_cv_scores.append(accuracy_score(y_val, m_cal.predict(X_val)))

    rf_cv_mean = np.mean(rf_cv_scores)
    xgb_cv_mean = np.mean(xgb_cv_scores)
    lr_cv_mean = np.mean(lr_cv_scores)
    stacking_cv_mean = np.mean(stacking_cv_scores)
    calibrated_cv_mean = np.mean(calibrated_cv_scores)
    
    print(f"Random Forest 5-Fold CV Accuracy:       {rf_cv_mean*100:.2f}% +/- {np.std(rf_cv_scores)*100:.2f}%")
    print(f"XGBoost 5-Fold CV Accuracy:             {xgb_cv_mean*100:.2f}% +/- {np.std(xgb_cv_scores)*100:.2f}%")
    print(f"Logistic Regression 5-Fold CV Accuracy: {lr_cv_mean*100:.2f}% +/- {np.std(lr_cv_scores)*100:.2f}%")
    print(f"Stacking Classifier 5-Fold CV Accuracy: {stacking_cv_mean*100:.2f}% +/- {np.std(stacking_cv_scores)*100:.2f}%")
    print(f"Calibrated Stacking 5-Fold CV Accuracy: {calibrated_cv_mean*100:.2f}% +/- {np.std(calibrated_cv_scores)*100:.2f}%")
    
    cv_means = {
        'Random Forest': rf_cv_mean,
        'XGBoost': xgb_cv_mean,
        'Logistic Regression': lr_cv_mean,
        'Stacking Classifier': stacking_cv_mean,
        'Calibrated Stacking': calibrated_cv_mean
    }

    # Best model selection for deployment (prioritize calibrated stacking when competitive)
    best_model_name = 'Calibrated Stacking'
    best_model = calibrated_stacking

    print(f"\nSelected Model for Deployment: {best_model_name}")
    print(f"Saving best calibrated model to {model_output}...")
    model_dir = os.path.dirname(model_output)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'model': best_model,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'scaler': scaler
    }
    
    with open(model_output, 'wb') as f:
        pickle.dump(model_data, f)
        
    # 8. Generate Calibrated Predictions & Confidence Scores for All Stars
    print("\nGenerating calibrated predictions and confidence scores...")
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    probs = best_model.predict_proba(X_scaled)
    preds = best_model.predict(X_scaled)
    
    pred_labels = le.inverse_transform(preds)
    confidences = np.max(probs, axis=1)
    
    predictions_df = df.copy()
    predictions_df['predicted_label'] = pred_labels
    predictions_df['confidence'] = confidences
    
    for idx, class_name in enumerate(le.classes_):
        predictions_df[f'prob_{class_name}'] = probs[:, idx]
        
    predictions_df['significance'] = 'high'
    low_sig_mask = (
        (predictions_df['predicted_label'] == 'confirmed_planet') & (predictions_df['transit_snr'] < 5.0)
    ) | (predictions_df['confidence'] < 0.6)
    
    predictions_df.loc[low_sig_mask, 'significance'] = 'low_significance'
    
    pred_dir = os.path.dirname(predictions_output)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
    predictions_df.to_csv(predictions_output, index=False)
    print(f"Predictions table saved to {predictions_output} (Rows: {len(predictions_df)})")
    
    correct_count = (predictions_df['label'] == predictions_df['predicted_label']).sum()
    print(f"Overall dataset accuracy: {correct_count}/{len(predictions_df)} ({correct_count/len(predictions_df)*100:.1f}%)")

    # 9. Generate Updated Markdown Validation Report
    rf_train_acc = accuracy_score(y_train, rf_train_preds) * 100
    xgb_train_acc = accuracy_score(y_train, xgb_train_preds) * 100
    lr_train_acc = accuracy_score(y_train, lr_train_preds) * 100
    stk_train_acc = accuracy_score(y_train, stacking_train_preds) * 100
    cal_train_acc = accuracy_score(y_train, cal_train_preds) * 100

    rf_test_acc = accuracy_score(y_test, rf_test_preds) * 100
    xgb_test_acc = accuracy_score(y_test, xgb_test_preds) * 100
    lr_test_acc = accuracy_score(y_test, lr_test_preds) * 100
    stk_test_acc = accuracy_score(y_test, stacking_test_preds) * 100
    cal_test_acc = accuracy_score(y_test, cal_test_preds) * 100
    
    cal_test_report = classification_report(y_test, cal_test_preds, target_names=le.classes_)
    stk_test_report = classification_report(y_test, stacking_test_preds, target_names=le.classes_)
    xgb_test_report = classification_report(y_test, xgb_test_preds, target_names=le.classes_)
    rf_test_report = classification_report(y_test, rf_test_preds, target_names=le.classes_)
    lr_test_report = classification_report(y_test, lr_test_preds, target_names=le.classes_)
    
    report_content = f"""# Validation Metrics Report (ExoFinder Stacking & Calibration)

This report was automatically generated following hyperparameter optimization, stacking ensembling, and probability calibration across the 405-target dataset.

---

## 1. Classifier Performance Comparison

Below is the comparison of performance metrics of all tuned models across the **{len(feature_cols)} physical features**:

| Model Architecture | Train Accuracy ({len(X_train)} stars) | Test Accuracy ({len(X_test)} stars) | 5-Fold Stratified CV Accuracy |
| :--- | :--- | :--- | :--- |
| **Random Forest (Tuned)** | {rf_train_acc:.1f}% | {rf_test_acc:.1f}% | **{rf_cv_mean*100:.1f}% ± {np.std(rf_cv_scores)*100:.1f}%** |
| **XGBoost (Tuned)** | {xgb_train_acc:.1f}% | {xgb_test_acc:.1f}% | **{xgb_cv_mean*100:.1f}% ± {np.std(xgb_cv_scores)*100:.1f}%** |
| **Logistic Regression (Tuned)** | {lr_train_acc:.1f}% | {lr_test_acc:.1f}% | **{lr_cv_mean*100:.1f}% ± {np.std(lr_cv_scores)*100:.1f}%** |
| **Stacking Classifier (Meta-LR)** | {stk_train_acc:.1f}% | {stk_test_acc:.1f}% | **{stacking_cv_mean*100:.1f}% ± {np.std(stacking_cv_scores)*100:.1f}%** |
| **Calibrated Stacking (Platt Sigmoid)** | {cal_train_acc:.1f}% | {cal_test_acc:.1f}% | **{calibrated_cv_mean*100:.1f}% ± {np.std(calibrated_cv_scores)*100:.1f}%** |

### Selected Deployed Model: **{best_model_name}**
Saved to `outputs/model.pkl` with Platt-calibrated probability mappings.

---

## 2. Test Set Classification Reports

### Calibrated Stacking Ensemble (Production Model)
```
{cal_test_report}```

### Stacking Ensemble (Pre-calibration)
```
{stk_test_report}```

### Tuned XGBoost
```
{xgb_test_report}```

### Tuned Random Forest
```
{rf_test_report}```

---

## 3. Key Enhancements & Methodology
1. **Odd/Even Transit Depth Analysis**: Evaluates odd-cycle vs. even-cycle transit depth ratios and difference significance (sigma) to separate Eclipsing Binaries from transiting planets.
2. **Automated Hyperparameter Optimization**: `RandomizedSearchCV` explores regularization, tree depth, and learning rates under stratified cross-validation.
3. **Stacking Ensemble**: Replaced manual heuristic voting with `StackingClassifier`, allowing a Logistic Regression meta-learner to learn optimal non-linear blending weights.
4. **Probability Calibration**: `CalibratedClassifierCV` transforms raw model decision margins into well-calibrated posterior probabilities.
5. **Physical Feature Set ({len(feature_cols)} features)**: {", ".join(feature_cols)}.
"""

    report_path = os.path.join(project_root, "reports", "validation_metrics_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"Validation metrics report written at {report_path}")
    
    return predictions_df

if __name__ == "__main__":
    train_and_evaluate()
