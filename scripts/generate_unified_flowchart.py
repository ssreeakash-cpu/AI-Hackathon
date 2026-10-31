"""
Script: generate_unified_flowchart.py
Description: Generates a publication-grade, high-resolution flowchart merging:
  1. Image Classification Sequence (CLAHE + GAN + Overfit Rectification)
  2. CSV / Tabular Classification Sequence (SMOTE + Pruning + Overfit Rectification)
Outputs:
  - docs/methodology_flowchart.png
  - docs/unified_methodology_flowchart.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def build_flowchart():
    os.makedirs("docs", exist_ok=True)
    
    # Set overall figure dimensions and styling
    fig = plt.figure(figsize=(18, 24), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background canvas: clean academic off-white / light slate
    bg = patches.Rectangle((0, 0), 100, 100, facecolor="#F8FAFC", edgecolor="none")
    ax.add_patch(bg)

    # Palette definition (academic, high contrast, zero emojis)
    c_title = "#0F172A"       # Slate 900
    c_subtitle = "#475569"    # Slate 600
    c_shared_bg = "#EDE9FE"   # Violet 100
    c_shared_edge = "#6D28D9" # Violet 700
    c_shared_txt = "#4C1D95"  # Violet 900
    
    c_img_bg = "#EFF6FF"      # Blue 50
    c_img_edge = "#1D4ED8"    # Blue 700
    c_img_txt = "#1E3A8A"     # Blue 900
    c_img_hdr = "#2563EB"     # Blue 600
    
    c_csv_bg = "#F0FDF4"      # Emerald 50
    c_csv_edge = "#047857"    # Emerald 700
    c_csv_txt = "#064E3B"     # Emerald 900
    c_csv_hdr = "#059669"     # Emerald 600

    c_arrow = "#334155"       # Slate 700

    # Header Section
    ax.text(50, 97.5, "Unified Dual-Branch Methodology & Verification Framework",
            ha="center", va="center", fontsize=24, fontweight="bold", color=c_title, fontfamily="sans-serif")
    ax.text(50, 95.8, "End-to-End Image Classification & CSV Tabular Imbalance Handling Pipeline with Leakage Auditing",
            ha="center", va="center", fontsize=13, color=c_subtitle, fontfamily="sans-serif")

    # Column Headers
    # Branch A: Image Modality
    col_a_box = FancyBboxPatch((5, 91.5), 42, 2.8, boxstyle="round,pad=0.3",
                               facecolor=c_img_hdr, edgecolor=c_img_edge, linewidth=1.5)
    ax.add_patch(col_a_box)
    ax.text(26, 92.9, "Branch A: Image Classification Stream",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    # Branch B: CSV Tabular Modality
    col_b_box = FancyBboxPatch((53, 91.5), 42, 2.8, boxstyle="round,pad=0.3",
                               facecolor=c_csv_hdr, edgecolor=c_csv_edge, linewidth=1.5)
    ax.add_patch(col_b_box)
    ax.text(74, 92.9, "Branch B: CSV Tabular Classification Stream",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    def draw_box(x, y, w, h, bg_col, edge_col, title, lines, title_col=None):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                             facecolor=bg_col, edgecolor=edge_col, linewidth=1.6)
        ax.add_patch(box)
        t_col = title_col if title_col else edge_col
        ax.text(x + w / 2, y + h - 0.9, title,
                ha="center", va="top", fontsize=11, fontweight="bold", color=t_col)
        
        spacing = (h - 1.8) / (len(lines) + 0.5) if lines else 0
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 1.9 - (i * spacing), line,
                    ha="center", va="top", fontsize=9.2, color="#1E293B")

    def draw_arrow(x1, y1, x2, y2, color=c_arrow):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle="-|>", mutation_scale=14,
                                color=color, linewidth=1.8, linestyle="solid")
        ax.add_patch(arrow)

    # -------------------------------------------------------------
    # STAGE 1: LOAD DATASET (y: 84 to 89.5, h=5.5)
    # -------------------------------------------------------------
    draw_box(5, 84, 42, 5.5, c_img_bg, c_img_edge,
             "1. Load Image Dataset",
             ["Ingest Multi-Source Ground & Satellite Imagery",
              "Classes: Fire, Smoke, Environmental Distractors (Clouds/Fog), Clear",
              "Catalog Verification: Dimensions, Integrity, and Channels"],
             c_img_txt)

    draw_box(53, 84, 42, 5.5, c_csv_bg, c_csv_edge,
             "1. Load CSV Dataset",
             ["Ingest NASA VIIRS / MODIS & Weather Telemetry",
              "Features: Brightness Temperature, FRP, Confidence, Weather Indices",
              "Catalog Verification: Record Counts, Missing Attributes, Types"],
             c_csv_txt)

    draw_arrow(26, 84, 26, 81.5)
    draw_arrow(74, 84, 74, 81.5)

    # -------------------------------------------------------------
    # STAGE 2: DATA PREPROCESSING (y: 76 to 81.5, h=5.5)
    # -------------------------------------------------------------
    draw_box(5, 76, 42, 5.5, c_img_bg, c_img_edge,
             "2. Image Preprocessing",
             ["Spatial Standardization: Resizing to 224 x 224 Pixels",
              "Tensor Conversion & RGB Channel Standardization",
              "Image Normalization via Domain Mean and Standard Deviation"],
             c_img_txt)

    draw_box(53, 76, 42, 5.5, c_csv_bg, c_csv_edge,
             "2. Tabular Preprocessing",
             ["Missing Value Imputation: Median & Mode Substitution",
              "Target Formulation: High-Severity Hazard vs. Baseline",
              "Categorical Encoding: One-Hot & Numerical Vectorization"],
             c_csv_txt)

    draw_arrow(26, 76, 35, 73.5)
    draw_arrow(74, 76, 65, 73.5)

    # -------------------------------------------------------------
    # STAGE 3: DATA LEAKAGE CHECKS (SHARED CENTER BOX, y: 67 to 73.5, h=6.5)
    # -------------------------------------------------------------
    draw_box(14, 67, 72, 6.5, c_shared_bg, c_shared_edge,
             "3. Rigorous Multi-Modal Data Leakage Checks",
             ["Target Leakage Audit: Correlation Checks with Ground Truth to Prevent Target Proxy Ingestion",
              "Duplicate Leakage Audit: MD5 Checksums & Perceptual Hashing (pHash) Between Partitions",
              "Suspicious Feature Diagnostic: Flag and Remove High-Cardinality Artifacts & Timestamp Leaks"],
             c_shared_txt)

    draw_arrow(26, 67, 26, 64.5)
    draw_arrow(74, 67, 74, 64.5)

    # -------------------------------------------------------------
    # STAGE 4: DOMAIN FEATURE TRANSFORMATION (y: 59 to 64.5, h=5.5)
    # -------------------------------------------------------------
    draw_box(5, 59, 42, 5.5, c_img_bg, c_img_edge,
             "4. Contrast Enhancement (CLAHE)",
             ["Contrast Limited Adaptive Histogram Equalization Applied to Luminance (L/Y)",
              "Enhances Low-Contrast Smoke Plumes in Diffuse & Hazy Atmospheric Scenes",
              "Prevents Local Noise Amplification via Dynamic Clip Limits (Clip=2.0)"],
             c_img_txt)

    draw_box(53, 59, 42, 5.5, c_csv_bg, c_csv_edge,
             "4. Feature Scaling & Numerical Normalization",
             ["StandardScaler Transformation: Zero Mean, Unit Variance",
              "Strict Protocol: Scaler is Fit ONLY on Training Data (Zero Test Leakage)",
              "Validation and Test Partitions Evaluated via Frozen Scaler Parameters"],
             c_csv_txt)

    draw_arrow(26, 59, 35, 56.5)
    draw_arrow(74, 59, 65, 56.5)

    # -------------------------------------------------------------
    # STAGE 5: DATASET PARTITIONING PROTOCOL (SHARED CENTER BOX, y: 50 to 56.5, h=6.5)
    # -------------------------------------------------------------
    draw_box(14, 50, 72, 6.5, c_shared_bg, c_shared_edge,
             "5. Dataset Partitioning & Cross-Validation Protocol",
             ["Protocol 1: Stratified Train / Validation / Test Split (70% Train, 15% Validation, 15% Test)",
              "Protocol 2: 10-Fold Stratified Cross-Validation (K=10) on Development Partition",
              "Class Distribution Preservation across All Folds with Independent Held-Out Test Isolation"],
             c_shared_txt)

    draw_arrow(26, 50, 26, 47.5)
    draw_arrow(74, 50, 74, 47.5)

    # -------------------------------------------------------------
    # STAGE 6: CLASS BALANCING ON TRAINING DATA ONLY (y: 41.5 to 47.5, h=6.0)
    # -------------------------------------------------------------
    draw_box(5, 41.5, 42, 6.0, c_img_bg, c_img_edge,
             "6. GAN Augmentation (TRAINING Data Only)",
             ["Deep Convolutional GAN (DCGAN) Architecture with Binary Cross-Entropy Loss",
              "Synthesizes Realistic Minority Class Samples (Active Flame Fronts & Smoke)",
              "STRICT LEAKAGE CONTROL: Validation & Test Partitions Remain Strictly Untouched"],
             c_img_txt)

    draw_box(53, 41.5, 42, 6.0, c_csv_bg, c_csv_edge,
             "6. SMOTE Balancing (TRAINING Data Only)",
             ["Synthetic Minority Over-sampling Technique via K-Nearest Neighbor Interpolation",
              "Balances Minority Severe Wildfire Records with Major Ambient Observations",
              "STRICT LEAKAGE CONTROL: Validation & Test Partitions Remain Strictly Untouched"],
             c_csv_txt)

    draw_arrow(26, 41.5, 26, 39.0)
    draw_arrow(74, 41.5, 74, 39.0)

    # -------------------------------------------------------------
    # STAGE 7: INITIAL MODEL TRAINING (y: 33.5 to 39.0, h=5.5)
    # -------------------------------------------------------------
    draw_box(5, 33.5, 42, 5.5, c_img_bg, c_img_edge,
             "7. Initial Training: 5 Hybrid Techniques",
             ["1. YOLO + EfficientNet + LSTM | 2. CNN + BiLSTM + Attention",
              "3. Swin Transformer + CNN | 4. YOLO + ViT + LSTM",
              "5. ConvLSTM + Attention (Spatio-Temporal Sequence Modeling)"],
             c_img_txt)

    draw_box(53, 33.5, 42, 5.5, c_csv_bg, c_csv_edge,
             "7. Initial Tabular Model Training",
             ["Baseline Random Forest Classifier (Unconstrained Tree Depth)",
              "Gini Impurity Criterion with 100 Estimators",
              "Trained Exclusively on SMOTE-Balanced Training Feature Vectors"],
             c_csv_txt)

    draw_arrow(26, 33.5, 35, 31.0)
    draw_arrow(74, 33.5, 65, 31.0)

    # -------------------------------------------------------------
    # STAGE 8: OVERFITTING / UNDERFITTING DIAGNOSIS (SHARED CENTER BOX, y: 24.5 to 31.0, h=6.5)
    # -------------------------------------------------------------
    draw_box(14, 24.5, 72, 6.5, c_shared_bg, c_shared_edge,
             "8. Overfitting / Underfitting Detection & Diagnostics",
             ["Learning Curve Analysis: Real-Time Tracking of Training Loss vs. Validation Loss",
              "Generalization Gap Audit: Quantifying Divergence Between Training & Validation Accuracy",
              "Diagnostic Triggers: Detected Memorization / High-Variance Overfitting Triggers Phase 9 Rectification"],
             c_shared_txt)

    draw_arrow(26, 24.5, 26, 22.0)
    draw_arrow(74, 24.5, 74, 22.0)

    # -------------------------------------------------------------
    # STAGE 9: APPLY CORRECTION TECHNIQUE & RETRAIN (y: 15.5 to 22.0, h=6.5)
    # -------------------------------------------------------------
    draw_box(5, 15.5, 42, 6.5, c_img_bg, c_img_edge,
             "9. Apply Correction Technique & Retrain",
             ["Spatial Dropout (p=0.5) to Interrupt Co-Adapted Feature Extractors",
              "Batch Normalization Layers for Activation Stabilization",
              "L2 Weight Decay (1e-4), Label Smoothing (eps=0.1), and Early Stopping Retraining"],
             c_img_txt)

    draw_box(53, 15.5, 42, 6.5, c_csv_bg, c_csv_edge,
             "9. Apply Correction Technique & Retrain",
             ["Structural Tree Pruning: Maximum Depth Constraints (max_depth=12)",
              "Node Splitting Constraints: min_samples_split=8, min_samples_leaf=4",
              "Transition to Regularized Gradient Boosting Classifier with Learning Rate Shrinkage"],
             c_csv_txt)

    draw_arrow(26, 15.5, 26, 13.0)
    draw_arrow(74, 15.5, 74, 13.0)

    # -------------------------------------------------------------
    # STAGE 10: FINAL TEST EVALUATION (y: 7.0 to 13.0, h=6.0)
    # -------------------------------------------------------------
    draw_box(5, 7.0, 42, 6.0, c_img_bg, c_img_edge,
             "10. Final Vision Test Evaluation",
             ["Evaluated Exclusively on Isolated, Unseen Test Split (Zero Information Leakage)",
              "Multi-Class Metrics: Accuracy, Precision, Recall, Macro F1-Score",
              "Confusion Matrix Analysis Against Confusing Distractor Benchmarks"],
             c_img_txt)

    draw_box(53, 7.0, 42, 6.0, c_csv_bg, c_csv_edge,
             "10. Final Tabular Test Evaluation",
             ["Evaluated Exclusively on Isolated, Unseen Test Split (Zero Information Leakage)",
              "Imbalance-Resilient Metrics: Balanced Accuracy, ROC-AUC, Macro F1-Score",
              "Brier Score Calibration Analysis & Confusion Matrix Verification"],
             c_csv_txt)

    draw_arrow(26, 7.0, 35, 4.8)
    draw_arrow(74, 7.0, 65, 4.8)

    # -------------------------------------------------------------
    # STAGE 11: MULTI-MODAL LATE FUSION (SHARED CENTER BOX, y: 0.8 to 4.8, h=4.0)
    # -------------------------------------------------------------
    draw_box(14, 0.8, 72, 4.0, c_shared_bg, c_shared_edge,
             "11. Output: Fire/Smoke Detection -> Severity -> Geolocation -> Alert",
             ["Multi-Modal Late-Fusion Risk Engine Combines Visual Predictions with Satellite FRP Telemetry",
              "Dispatches Early-Warning Alerts with Location, Severity, Confidence Score & Estimated Affected Area"],
             c_shared_txt)

    # Save to unified target location
    path1 = os.path.join("docs", "unified_methodology_flowchart.png")
    
    plt.savefig(path1, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    print(f"[OK] Generated: {path1}")

if __name__ == "__main__":
    build_flowchart()
