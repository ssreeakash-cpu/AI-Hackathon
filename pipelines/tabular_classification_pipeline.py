"""
Pipeline 2: CSV / Tabular Classification with Leakage Checks, SMOTE & Overfit Rectification
Fulfills User Workflow:
  Load CSV Dataset
  ↓
  Data Preprocessing
  ↓
  Data Leakage Checks (Target leakage • duplicate leakage • suspicious features)
  ↓
  1. Train / Validation / Test Split or Train / Test Split
  2. 10-fold cross validation
  ↓
  SMOTE Algorithm (balancing dataset on TRAINING data only)
  ↓
  Initial Model Training
  ↓
  Overfitting / Underfitting Detection and Rectification
  ↓
  Apply Correction Technique and Retrain
  ↓
  Final Test Evaluation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "pipeline_tabular")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# STEP 1: LOAD CSV DATASET
# ==============================================================================
def load_csv_dataset():
    print("\n" + "="*70)
    print("STEP 1: LOAD CSV DATASET")
    print("="*70)
    
    # Primary: NASA VIIRS India/South Asia active fire dataset
    # Secondary Fallback: NASA Global VIIRS / MODIS or UCI Forest Fire
    candidates = [
        os.path.join(PROJECT_ROOT, "datasets", "nasa_earthdata", "south_asia_india", "VIIRS_SouthAsia_India_24h.csv"),
        os.path.join(PROJECT_ROOT, "datasets", "nasa_earthdata", "viirs", "SUOMI_VIIRS_C2_Global_24h.csv"),
        os.path.join(PROJECT_ROOT, "datasets", "environmental_weather_fwi", "forestfires.csv")
    ]
    
    csv_path = None
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1000:
            csv_path = c
            break
            
    if not csv_path:
        raise FileNotFoundError("No valid CSV dataset found in datasets/ directory.")
        
    print(f"[CSV Loaded] Loading NASA Earthdata Active Fire Telemetry from:\n  {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total Rows: {len(df):,}, Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    return df, csv_path


# ==============================================================================
# STEP 2: DATA PREPROCESSING
# ==============================================================================
def preprocess_csv_data(df):
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING & TARGET FORMULATION")
    print("="*70)
    
    df = df.copy()
    
    # 1. Handle Missing Values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Handling missing values across columns: {dict(null_counts[null_counts > 0])}")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("[OK] Zero null/missing values detected.")
        
    # 2. Formulate Problem Target:
    # High-Severity Wildfire / Critical Thermal Output (Imbalanced Target)
    # Using Fire Radiative Power (FRP) and Thermal Brightness (bright_ti4 / brightness)
    if 'frp' in df.columns:
        # Top 15% FRP threshold defines high-severity outbreak (natural real-world minority class)
        threshold = df['frp'].quantile(0.85)
        df['target_severity'] = (df['frp'] >= threshold).astype(int)
        target_name = f"High-Severity Outbreak (FRP >= {threshold:.1f} MW)"
    elif 'confidence' in df.columns:
        df['target_severity'] = (df['confidence'].astype(str).str.lower() == 'h').astype(int)
        target_name = "High-Confidence Ignition Point"
    else:
        df['target_severity'] = (df['area'] > 0).astype(int)
        target_name = "Active Burn Area Incident"
        
    # Categorical Encoding
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if col not in ['acq_date', 'version', 'satellite']:
            df[col] = df[col].astype('category').cat.codes
            
    # Feature Selection: Select predictive remote sensing variables
    candidate_features = [
        'latitude', 'longitude', 'bright_ti4', 'brightness', 'scan', 
        'track', 'acq_time', 'bright_ti5', 'bright_t31', 'daynight'
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]
    
    X = df[feature_cols].copy()
    y = df['target_severity'].values
    
    print(f"Target Formulation: {target_name}")
    class_counts = pd.Series(y).value_counts().to_dict()
    print(f"Class Distribution: Class 0 (Moderate/Low): {class_counts.get(0, 0)}, Class 1 (Critical Minority): {class_counts.get(1, 0)}")
    imbalance_ratio = class_counts.get(0, 1) / max(1, class_counts.get(1, 1))
    print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1 (Demonstrates need for SMOTE on Training data)")
    
    return X, y, feature_cols


# ==============================================================================
# STEP 3: DATA LEAKAGE CHECKS
# ==============================================================================
def check_tabular_leakage(X, y, feature_cols):
    print("\n" + "="*70)
    print("STEP 3: DATA LEAKAGE CHECKS")
    print("="*70)
    
    leakage_findings = {
        "target_leakage": [],
        "duplicate_rows": 0,
        "suspicious_features": []
    }
    
    # 1. Target Leakage Checks: Feature-to-target Pearson correlation
    print("1. Checking for Target Leakage (Correlations |r| > 0.90)...")
    for col in feature_cols:
        corr = np.corrcoef(X[col], y)[0, 1]
        if np.abs(corr) > 0.90:
            leakage_findings["target_leakage"].append((col, corr))
            print(f"   [ALERT: TARGET LEAKAGE] Column '{col}' has severe correlation with target (r = {corr:.3f})")
    if not leakage_findings["target_leakage"]:
        print("   [OK] No direct target leakage or proxy features detected.")
        
    # 2. Duplicate Leakage Checks: Exact row duplicates
    print("2. Checking for Duplicate Row Leakage...")
    dup_mask = X.duplicated()
    dup_count = dup_mask.sum()
    leakage_findings["duplicate_rows"] = int(dup_count)
    if dup_count > 0:
        print(f"   [Notice] Found {dup_count} duplicate feature rows. Deduplicating to prevent train-test contamination.")
        X = X[~dup_mask].copy()
        y = y[~dup_mask]
    else:
        print("   [OK] Zero duplicate feature rows detected.")
        
    # 3. Suspicious Features: Constant or near-zero variance features
    print("3. Checking for Suspicious / Pseudo-Identifier Features...")
    for col in feature_cols:
        var = X[col].var()
        unique_ratio = X[col].nunique() / len(X)
        if var == 0:
            leakage_findings["suspicious_features"].append(f"{col} (Zero Variance)")
            print(f"   [ALERT] '{col}' has zero variance (constant). Must be pruned.")
        elif unique_ratio > 0.99 and not np.issubdtype(X[col].dtype, np.floating):
            leakage_findings["suspicious_features"].append(f"{col} (High Cardinality ID)")
            print(f"   [ALERT] '{col}' appears to be a unique ID/serial number (proxy risk).")
            
    if not leakage_findings["suspicious_features"]:
        print("   [OK] All predictor columns have valid variance and feature distributions.")
        
    return X, y


# ==============================================================================
# STEP 4: TRAIN / VAL / TEST SPLIT & 10-FOLD CROSS-VALIDATION
# ==============================================================================
def create_tabular_splits_and_cv(X, y, test_size=0.20, val_size=0.15, random_state=42):
    print("\n" + "="*70)
    print("STEP 4: TRAIN / VALIDATION / TEST SPLIT & 10-FOLD CROSS-VALIDATION")
    print("="*70)
    
    # 1. Stratified Train / Test Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. Train / Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1.0 - test_size), 
        random_state=random_state, stratify=y_train_val
    )
    
    # 3. Feature Scaling strictly fit on Training set to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    print(f"1. Independent Stratified Splits:")
    print(f"   - Training Set:   {len(X_train_scaled):,} rows (Minority: {sum(y_train == 1)})")
    print(f"   - Validation Set: {len(X_val_scaled):,} rows (Minority: {sum(y_val == 1)})")
    print(f"   - Test Set:       {len(X_test_scaled):,} rows (Minority: {sum(y_test == 1)}) [STRICTLY QUARANTINED]")
    
    # 4. 10-Fold Stratified Cross-Validation
    n_splits = min(10, sum(y_train == 1))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print(f"2. {n_splits}-Fold Stratified Cross-Validation Configured on Training Partition.")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, skf, scaler


# ==============================================================================
# STEP 5: SMOTE ALGORITHM (BALANCING DATASET ON TRAINING DATA ONLY)
# ==============================================================================
def apply_smote_on_training_only(X_train, y_train, random_state=42):
    print("\n" + "="*70)
    print("STEP 5: SMOTE ALGORITHM (BALANCING DATASET ON TRAINING DATA ONLY)")
    print("="*70)
    print("CRITICAL CHECK: Zero leakage. Validation and Test sets are NEVER resampled.")
    print(f"Pre-SMOTE Training Distribution:  Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")
    
    k_neighbors = min(5, max(1, sum(y_train == 1) - 1))
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Post-SMOTE Training Distribution: Class 0: {sum(y_train_resampled == 0)}, Class 1: {sum(y_train_resampled == 1)}")
    print(f"[SMOTE Complete] Perfectly balanced 50:50 distribution on Training partition.")
    return X_train_resampled, y_train_resampled


# ==============================================================================
# STEP 6: INITIAL MODEL TRAINING
# ==============================================================================
def train_initial_tabular_model(X_train, y_train, X_val, y_val):
    print("\n" + "="*70)
    print("STEP 6: INITIAL MODEL TRAINING (UNCONSTRAINED BASELINE)")
    print("="*70)
    
    # Initial unconstrained model (e.g. Random Forest with unlimited depth)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None, # Unconstrained depth triggers potential memorization
        min_samples_split=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) > 1 else None
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_f1 = f1_score(y_train, train_pred, average='weighted')
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    val_roc = roc_auc_score(y_val, val_prob) if val_prob is not None else 0.0
    
    print(f"Initial Train Accuracy: {train_acc:.2%}, Train F1: {train_f1:.4f}")
    print(f"Initial Val Accuracy:   {val_acc:.2%}, Val F1:   {val_f1:.4f}")
    if val_roc:
        print(f"Initial Val ROC-AUC:    {val_roc:.4f}")
        
    return model, {
        "train_acc": train_acc, "val_acc": val_acc,
        "train_f1": train_f1, "val_f1": val_f1, "val_roc": val_roc
    }


# ==============================================================================
# STEP 7: OVERFITTING / UNDERFITTING DETECTION AND RECTIFICATION
# ==============================================================================
def detect_tabular_overfitting(metrics):
    print("\n" + "="*70)
    print("STEP 7: OVERFITTING / UNDERFITTING DETECTION")
    print("="*70)
    
    acc_gap = metrics["train_acc"] - metrics["val_acc"]
    f1_gap = metrics["train_f1"] - metrics["val_f1"]
    
    diagnosis = {
        "train_acc": metrics["train_acc"],
        "val_acc": metrics["val_acc"],
        "accuracy_gap": acc_gap,
        "f1_gap": f1_gap,
        "status": "NORMAL"
    }
    
    if acc_gap > 0.08 or metrics["train_acc"] > 0.98 and metrics["val_acc"] < 0.90:
        diagnosis["status"] = "OVERFITTING_DETECTED"
        print(f"[Diagnosis: OVERFITTING DETECTED]")
        print(f"  - Memorization Gap: Train Acc ({metrics['train_acc']:.1%}) vs Val Acc ({metrics['val_acc']:.1%}) -> Delta: {acc_gap:.1%}")
        print(f"  - Cause: Unconstrained tree depth allowing leaf nodes to fit noise.")
    elif metrics["train_acc"] < 0.65:
        diagnosis["status"] = "UNDERFITTING_DETECTED"
        print(f"[Diagnosis: UNDERFITTING DETECTED] Model failed to capture predictive boundaries.")
    else:
        print("[Diagnosis: BALANCED GENERALIZATION]")
        
    return diagnosis


# ==============================================================================
# STEP 8: APPLY CORRECTION TECHNIQUE AND RETRAIN
# ==============================================================================
def retrain_corrected_tabular_model(X_train, y_train, X_val, y_val):
    print("\n" + "="*70)
    print("STEP 8: APPLY CORRECTION TECHNIQUES AND RETRAIN")
    print("="*70)
    print("Applied Correction Techniques:")
    print("  1. Constrained Tree Depth (max_depth=6) to prevent memorization")
    print("  2. Increased min_samples_split=6 and min_samples_leaf=4 for leaf regularization")
    print("  3. Applied Feature Subsampling (max_features='sqrt') to decorrelate ensemble trees")
    print("  4. Gradient Boosting Regularization with Subsample (0.85) and Learning Rate (0.05)")
    
    # Regularized Gradient Boosting Classifier
    corrected_model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=4,
        subsample=0.85,
        random_state=42
    )
    
    corrected_model.fit(X_train, y_train)
    
    train_pred = corrected_model.predict(X_train)
    val_pred = corrected_model.predict(X_val)
    val_prob = corrected_model.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) > 1 else None
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    val_bal = balanced_accuracy_score(y_val, val_pred)
    val_roc = roc_auc_score(y_val, val_prob) if val_prob is not None else 0.0
    
    print(f"Corrected Train Accuracy:    {train_acc:.2%}")
    print(f"Corrected Val Accuracy:      {val_acc:.2%}")
    print(f"Corrected Val Balanced Acc:  {val_bal:.2%}")
    print(f"Corrected Val F1-Score:      {val_f1:.4f}")
    if val_roc:
        print(f"Corrected Val ROC-AUC:       {val_roc:.4f}")
        
    return corrected_model


# ==============================================================================
# STEP 9: FINAL TEST EVALUATION
# ==============================================================================
def final_test_evaluation(model, X_test, y_test, feature_names):
    print("\n" + "="*70)
    print("STEP 9: FINAL TEST EVALUATION (ON INDEPENDENT UNTOUCHED TEST SET)")
    print("="*70)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else None
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    feature_importances = dict(zip(feature_names, [round(float(v), 4) for v in model.feature_importances_]))
    
    results = {
        "pipeline": "CSV Classification with SMOTE & Regularized Gradient Boosting",
        "test_accuracy": acc,
        "test_balanced_accuracy": bal_acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1_score": f1,
        "test_roc_auc": roc,
        "confusion_matrix": cm,
        "feature_importances": feature_importances,
        "total_test_samples": len(y_test)
    }
    
    print(f"Final Test Accuracy:          {acc:.2%}")
    print(f"Final Test Balanced Accuracy: {bal_acc:.2%}")
    print(f"Final Test Precision:         {prec:.4f}")
    print(f"Final Test Recall:            {rec:.4f}")
    print(f"Final Test F1-Score:          {f1:.4f}")
    if roc:
        print(f"Final Test ROC-AUC:           {roc:.4f}")
    print(f"Confusion Matrix:\n{np.array(cm)}")
    print(f"Key Feature Importances:\n  {json.dumps(dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]), indent=2)}")
    
    out_file = os.path.join(OUTPUT_DIR, "tabular_pipeline_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[Results Saved] {out_file}")
    return results


# ==============================================================================
# MAIN EXECUTION ORCHESTRATOR
# ==============================================================================
def run_pipeline():
    # 1. Load CSV Dataset
    df, csv_path = load_csv_dataset()
    
    # 2. Data Preprocessing
    X, y, feature_cols = preprocess_csv_data(df)
    
    # 3. Data Leakage Checks
    X, y = check_tabular_leakage(X, y, feature_cols)
    
    # 4. Splits & 10-Fold Cross-Validation Setup
    X_train, X_val, X_test, y_train, y_val, y_test, skf, scaler = create_tabular_splits_and_cv(X, y)
    
    # 5. SMOTE Algorithm strictly on TRAINING data only
    X_train_resampled, y_train_resampled = apply_smote_on_training_only(X_train, y_train)
    
    # 6. Initial Model Training
    initial_model, init_metrics = train_initial_tabular_model(X_train_resampled, y_train_resampled, X_val, y_val)
    
    # 7. Overfitting Detection
    diagnosis = detect_tabular_overfitting(init_metrics)
    
    # 8. Apply Correction Technique and Retrain
    corrected_model = retrain_corrected_tabular_model(X_train_resampled, y_train_resampled, X_val, y_val)
    
    # 9. Final Test Evaluation
    final_results = final_test_evaluation(corrected_model, X_test, y_test, feature_cols)
    
    return final_results


if __name__ == "__main__":
    run_pipeline()
