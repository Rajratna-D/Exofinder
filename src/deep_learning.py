import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def generate_binned_views(time, flux, period, epoch, num_global_bins=201, num_local_bins=61, local_phase_width=0.08):
    """
    Bins detrended light curve into two multi-scale representations:
    1. Global View (201 bins): captures full orbital baseline [-0.5, 0.5]
    2. Local Zoomed View (61 bins): zoomed transit window [-local_phase_width, local_phase_width]
    
    Both views are zero-centered (flux - 1.0) so baseline is 0.0 and transits dip negative.
    """
    if len(time) == 0 or np.isnan(period) or period <= 0:
        return np.zeros(num_global_bins, dtype=np.float32), np.zeros(num_local_bins, dtype=np.float32)
        
    # Phase fold
    phase = ((time - epoch) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    
    # 1. Global View (-0.5 to 0.5)
    global_bins = np.linspace(-0.5, 0.5, num_global_bins + 1)
    global_view = np.zeros(num_global_bins, dtype=np.float32)
    for i in range(num_global_bins):
        mask = (phase >= global_bins[i]) & (phase < global_bins[i + 1])
        if np.any(mask):
            global_view[i] = float(np.nanmedian(flux[mask])) - 1.0
        else:
            global_view[i] = 0.0  # Normalized baseline
            
    # 2. Local Zoomed View (-local_phase_width to local_phase_width)
    local_bins = np.linspace(-local_phase_width, local_phase_width, num_local_bins + 1)
    local_view = np.zeros(num_local_bins, dtype=np.float32)
    for i in range(num_local_bins):
        mask = (phase >= local_bins[i]) & (phase < local_bins[i + 1])
        if np.any(mask):
            local_view[i] = float(np.nanmedian(flux[mask])) - 1.0
        else:
            local_view[i] = 0.0
            
    # Clean any accidental NaNs
    global_view = np.nan_to_num(global_view, nan=0.0)
    local_view = np.nan_to_num(local_view, nan=0.0)
    
    return global_view, local_view

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    torch = None
    nn = None
    optim = None
    Dataset = object
    DataLoader = None
    DEVICE = 'cpu'


class ExoDataset(Dataset):
    """
    PyTorch multi-modal Dataset for Exoplanet Detection with dynamic astronomical augmentation.
    """
    def __init__(self, global_views, local_views, tabular_features, labels=None, augment=False):
        self.global_views = np.asarray(global_views, dtype=np.float32)
        self.local_views = np.asarray(local_views, dtype=np.float32)
        self.tabular_features = np.asarray(tabular_features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64) if labels is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.global_views)

    def __getitem__(self, idx):
        g = self.global_views[idx].copy()
        l = self.local_views[idx].copy()
        t = self.tabular_features[idx].copy()
        
        # Astronomical dynamic augmentation for training
        if self.augment:
            # 1. Gaussian photon noise injection
            noise_g = np.random.normal(0, 0.0008, size=g.shape).astype(np.float32)
            noise_l = np.random.normal(0, 0.0008, size=l.shape).astype(np.float32)
            g = g + noise_g
            l = l + noise_l
            
            # 2. Depth scaling (+/- 5%)
            scale = np.random.uniform(0.95, 1.05)
            g = g * scale
            l = l * scale
            
            # 3. Small random horizontal jitter (roll array by +/- 1 bin)
            jitter_g = np.random.randint(-1, 2)
            jitter_l = np.random.randint(-1, 2)
            g = np.roll(g, jitter_g)
            l = np.roll(l, jitter_l)

        # PyTorch 1D-CNN expects shape: (channels=1, length)
        tensor_g = torch.tensor(g, dtype=torch.float32).unsqueeze(0)
        tensor_l = torch.tensor(l, dtype=torch.float32).unsqueeze(0)
        tensor_t = torch.tensor(t, dtype=torch.float32)
        
        if self.labels is not None:
            tensor_y = torch.tensor(self.labels[idx], dtype=torch.long)
            return tensor_g, tensor_l, tensor_t, tensor_y
        return tensor_g, tensor_l, tensor_t


class ExoFinderNet(nn.Module if nn is not None else object):
    """
    NASA Astronet-Inspired Dual-Branch 1D-CNN with Multi-Modal Tabular Feature Fusion.
    """
    def __init__(self, num_tabular_features=14, num_classes=3):
        if nn is None:
            raise ImportError("PyTorch is required to instantiate ExoFinderNet. Run 'pip install torch'.")
        super().__init__()
        
        # Branch 1: Global View 1D-CNN (201 bins)
        self.global_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 100
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 50
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),  # -> 64 * 10 = 640
            nn.Flatten()
        )
        
        # Branch 2: Local Zoomed View 1D-CNN (61 bins)
        self.local_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 30
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 15
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),  # -> 64 * 10 = 640
            nn.Flatten()
        )
        
        # Branch 3: Tabular Feature Dense Net
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-Modal Fusion Classification Head
        # Total combined features: 640 (global) + 640 (local) + 32 (tabular) = 1312
        self.classifier = nn.Sequential(
            nn.Linear(640 + 640 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_global, x_local, x_tab):
        feat_global = self.global_branch(x_global)
        feat_local = self.local_branch(x_local)
        feat_tab = self.tabular_branch(x_tab)
        
        fused = torch.cat([feat_global, feat_local, feat_tab], dim=1)
        logits = self.classifier(fused)
        return logits


def train_single_fold(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=45):
    """Trains a model on a single cross-validation fold and returns best validation accuracy."""
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for g, l, t, y in train_loader:
            g, l, t, y = g.to(DEVICE), l.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(g, l, t)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
        scheduler.step()
        
        # Evaluate on validation fold
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for g, l, t, y in val_loader:
                g, l, t, y = g.to(DEVICE), l.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
                logits = model(g, l, t)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    return best_val_acc, best_state


def train_and_evaluate_cnn(
    labels_csv=None,
    detrended_dir=None,
    bls_results_csv=None,
    features_csv=None,
    output_model_path=None,
    predictions_output=None,
    epochs=50,
    batch_size=32,
    lr=0.0015
):
    """
    End-to-end training and validation of the Dual-Branch 1D-CNN model.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if labels_csv is None:
        labels_csv = os.path.join(project_root, "data", "labels.csv")
    if detrended_dir is None:
        detrended_dir = os.path.join(project_root, "data", "detrended")
    if bls_results_csv is None:
        bls_results_csv = os.path.join(project_root, "outputs", "period_search_results.csv")
    if features_csv is None:
        features_csv = os.path.join(project_root, "outputs", "features.csv")
    if output_model_path is None:
        output_model_path = os.path.join(project_root, "outputs", "cnn_model.pt")
    if predictions_output is None:
        predictions_output = os.path.join(project_root, "outputs", "predictions.csv")

    print("\n==========================================")
    print("🪐 NASA Astronet Dual-Branch 1D-CNN Training")
    print(f"Compute Device: {DEVICE}")
    print("==========================================")

    # 1. Load Data
    df_labels = pd.read_csv(labels_csv)
    df_bls = pd.read_csv(bls_results_csv)
    df_feat = pd.read_csv(features_csv)

    feature_cols = [
        'bls_depth', 'bls_duration', 'bls_power', 
        'transit_snr', 'mean_to_max_depth_ratio', 'in_transit_skew', 
        'in_transit_kurtosis', 'flux_kurtosis',
        'ingress_egress_symmetry', 'secondary_eclipse_check',
        'period_ratio', 'depth_to_noise',
        'odd_even_depth_ratio', 'odd_even_mismatch_sigma'
    ]
    # Ensure derived features exist
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0

    print(f"Generating 1D-CNN Global & Local Binned Views for {len(df_feat)} targets...")
    global_views_list = []
    local_views_list = []
    valid_indices = []

    for idx, row in df_feat.iterrows():
        star_id = int(row['star_id'])
        det_file = os.path.join(detrended_dir, f"TIC_{star_id}.csv")
        
        bls_sub = df_bls[df_bls['star_id'] == star_id]
        if not os.path.exists(det_file) or len(bls_sub) == 0:
            continue
            
        bls_row = bls_sub.iloc[0]
        period = float(bls_row['recovered_period'])
        epoch = float(bls_row['recovered_epoch'])
        
        det_df = pd.read_csv(det_file)
        t_arr = det_df['time'].values
        f_arr = det_df['detrended_flux'].values
        
        g_view, l_view = generate_binned_views(t_arr, f_arr, period, epoch)
        global_views_list.append(g_view)
        local_views_list.append(l_view)
        valid_indices.append(idx)

    global_views = np.array(global_views_list, dtype=np.float32)
    local_views = np.array(local_views_list, dtype=np.float32)
    df_feat_valid = df_feat.iloc[valid_indices].reset_index(drop=True)

    X_tab = df_feat_valid[feature_cols].values
    y_raw = df_feat_valid['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"Generated views successfully. Global Shape: {global_views.shape}, Local Shape: {local_views.shape}")
    print(f"Class Distribution: {pd.Series(y_raw).value_counts().to_dict()}")

    # 2. Stratified Train / Test Split (80/20)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=42, stratify=y
    )

    X_g_train, X_g_test = global_views[train_idx], global_views[test_idx]
    X_l_train, X_l_test = local_views[train_idx], local_views[test_idx]
    X_t_train, X_t_test = X_tab[train_idx], X_tab[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize tabular features
    scaler = StandardScaler()
    X_t_train_scaled = scaler.fit_transform(X_t_train)
    X_t_test_scaled = scaler.transform(X_t_test)

    # Calculate class weights to balance cross entropy loss
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(len(y_train) / (len(class_counts) * class_counts), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. 5-Fold Stratified Cross-Validation on Training Split
    print("\n=== Running 5-Fold Stratified Cross-Validation ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (f_tr_idx, f_val_idx) in enumerate(skf.split(X_g_train, y_train)):
        ds_tr = ExoDataset(
            X_g_train[f_tr_idx], X_l_train[f_tr_idx], X_t_train_scaled[f_tr_idx], y_train[f_tr_idx], augment=True
        )
        ds_val = ExoDataset(
            X_g_train[f_val_idx], X_l_train[f_val_idx], X_t_train_scaled[f_val_idx], y_train[f_val_idx], augment=False
        )
        
        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
        
        model_fold = ExoFinderNet(num_tabular_features=len(feature_cols), num_classes=len(le.classes_)).to(DEVICE)
        opt_fold = optim.AdamW(model_fold.parameters(), lr=lr, weight_decay=1e-4)
        sched_fold = optim.lr_scheduler.CosineAnnealingLR(opt_fold, T_max=epochs, eta_min=1e-5)
        
        fold_acc, _ = train_single_fold(
            model_fold, loader_tr, loader_val, opt_fold, sched_fold, criterion, epochs=epochs
        )
        cv_scores.append(fold_acc)
        print(f"  Fold {fold + 1}/5 Accuracy: {fold_acc*100:.2f}%")

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\n1D-CNN 5-Fold CV Accuracy: {cv_mean*100:.2f}% +/- {cv_std*100:.2f}%")

    # 4. Train Final Model on Full Training Set
    print("\n=== Training Final 1D-CNN Production Model ===")
    final_ds_train = ExoDataset(X_g_train, X_l_train, X_t_train_scaled, y_train, augment=True)
    final_ds_test = ExoDataset(X_g_test, X_l_test, X_t_test_scaled, y_test, augment=False)
    
    loader_train = DataLoader(final_ds_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(final_ds_test, batch_size=batch_size, shuffle=False)
    
    final_model = ExoFinderNet(num_tabular_features=len(feature_cols), num_classes=len(le.classes_)).to(DEVICE)
    optimizer = optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    for epoch in range(epochs):
        final_model.train()
        for g, l, t, y_b in loader_train:
            g, l, t, y_b = g.to(DEVICE), l.to(DEVICE), t.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(final_model(g, l, t), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()

    # 5. Evaluate on Holdout Test Set
    final_model.eval()
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for g, l, t, y_b in loader_test:
            g, l, t = g.to(DEVICE), l.to(DEVICE), t.to(DEVICE)
            logits = final_model(g, l, t)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y_b.numpy())
            test_probs.extend(probs.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_report = classification_report(test_targets, test_preds, target_names=le.classes_)
    print(f"\n1D-CNN Test Set Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report (Test Set):")
    print(test_report)

    # 6. Save Model Checkpoint
    checkpoint = {
        'state_dict': final_model.state_dict(),
        'label_encoder': le,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'test_acc': test_acc
    }
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(checkpoint, output_model_path)
    print(f"Model saved successfully to {output_model_path}")

    # 7. Generate Full Dataset Predictions & Append CNN Probabilities to predictions.csv
    print("\nGenerating 1D-CNN predictions for all targets...")
    all_scaled_tab = scaler.transform(X_tab)
    full_ds = ExoDataset(global_views, local_views, all_scaled_tab, augment=False)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for g, l, t in full_loader:
            g, l, t = g.to(DEVICE), l.to(DEVICE), t.to(DEVICE)
            logits = final_model(g, l, t)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)
    all_confidences = np.max(all_probs, axis=1)
    all_pred_labels = le.inverse_transform(all_preds)

    # Merge into existing predictions.csv or create fresh
    if os.path.exists(predictions_output):
        preds_df = pd.read_csv(predictions_output)
    else:
        preds_df = df_feat_valid.copy()

    preds_df['predicted_label_cnn'] = all_pred_labels
    preds_df['confidence_cnn'] = all_confidences
    for i, class_name in enumerate(le.classes_):
        preds_df[f'prob_cnn_{class_name}'] = all_probs[:, i]

    preds_df.to_csv(predictions_output, index=False)
    print(f"Updated predictions saved to {predictions_output} (Rows: {len(preds_df)})")

    # 8. Write Markdown Metrics Report
    report_content = f"""# NASA Astronet 1D-CNN Deep Learning Report

This report summarizes the performance of the **Dual-Branch 1D-Convolutional Neural Network (1D-CNN)** with multi-modal tabular fusion trained on NASA TESS stellar light curves.

---

## 1. 1D-CNN Model Architecture

- **Global 1D-CNN Branch (201 bins)**: Multi-scale 1D convolutions ($k=5$, channels $1 \to 16 \to 32 \to 64$) extracting broad orbital context and secondary eclipse morphology.
- **Local Zoomed 1D-CNN Branch (61 bins)**: Fine-scale 1D convolutions ($k=3$, channels $1 \to 16 \to 32 \to 64$) resolving ingress/egress slopes and bottom curvature (U vs. V shape).
- **Tabular Feature Branch (14 features)**: Dense network ($14 \to 32$) with Batch Normalization and Dropout.
- **Multi-Modal Fusion**: Concatenates $640 + 640 + 32 = 1312$ features into dense classification layers with Dropout.

---

## 2. Validation Metrics

| Metric | Score |
| :--- | :--- |
| **5-Fold Stratified Cross-Validation** | **{cv_mean*100:.2f}% ± {cv_std*100:.2f}%** |
| **Holdout Test Set Accuracy** (81 stars) | **{test_acc*100:.2f}%** |

### Test Set Classification Report
```
{test_report}
```

---

## 3. Key Observations
1. **End-to-End Shape Perception**: The 1D-CNN directly captures the continuous curvature of planetary transit bottoms (U-shape) vs. binary star V-shapes without relying solely on summary metrics.
2. **Multi-Scale Context**: Combining the 201-bin global orbital baseline with the 61-bin zoomed transit window allows the network to simultaneously vet against out-of-transit stellar variability and evaluate in-transit ingress slopes.
"""
    report_path = os.path.join(project_root, "reports", "cnn_metrics_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"CNN metrics report written to {report_path}")
    
    return final_model

if __name__ == "__main__":
    train_and_evaluate_cnn()
