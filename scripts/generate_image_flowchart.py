"""
Script: generate_image_flowchart.py
Description: Generates a publication-grade, high-resolution flowchart for the
Image Classification Pipeline featuring the 5 Hybrid Techniques from Problem Statement 25:
  1. YOLO + EfficientNet + LSTM
  2. CNN + BiLSTM + Attention
  3. Swin Transformer + CNN
  4. YOLO + Vision Transformer + LSTM
  5. ConvLSTM + Attention
Output: Fire/smoke detection -> severity estimation -> geographic location -> early-warning alert.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def build_image_flowchart():
    os.makedirs("docs", exist_ok=True)
    
    # Dimensions: 20 x 28 inches at 300 DPI for high definition and readability
    fig = plt.figure(figsize=(20, 28), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background canvas: clean academic off-white / light slate
    bg = patches.Rectangle((0, 0), 100, 100, facecolor="#F8FAFC", edgecolor="none")
    ax.add_patch(bg)

    # Color Palette (Academic, High Contrast, Zero Emojis)
    c_title = "#0F172A"       # Slate 900
    c_subtitle = "#334155"    # Slate 700
    c_box_bg = "#FFFFFF"      # Crisp White
    c_box_edge = "#1E3A8A"    # Deep Navy Blue
    c_text_title = "#0F172A"  # Slate 900
    c_text_body = "#1E293B"   # Slate 800
    c_arrow = "#0F172A"       # Slate 900
    
    # Accent colors for functional stages
    c_stage_num = "#1D4ED8"   # Royal Blue
    c_hybrid_bg = "#EFF6FF"   # Light Blue tint
    c_hybrid_edge = "#2563EB" # Bright Blue
    c_leak_bg = "#FEF2F2"     # Soft Red tint
    c_leak_edge = "#B91C1C"   # Deep Red
    c_out_bg = "#F0FDF4"      # Soft Green tint
    c_out_edge = "#15803D"    # Deep Forest Green

    # Header Banner
    header_box = FancyBboxPatch((4, 93.8), 92, 5.0, boxstyle="round,pad=0.4",
                                facecolor="#0F172A", edgecolor="#1E293B", linewidth=2.0)
    ax.add_patch(header_box)
    
    ax.text(50, 97.0, "AI-BASED EARLY DETECTION OF FOREST FIRE AND SMOKE",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#FFFFFF", fontfamily="sans-serif")
    ax.text(50, 95.0, "Problem Statement 25 • Image Classification Pipeline & 5 Hybrid Techniques Architecture",
            ha="center", va="center", fontsize=12, fontweight="medium", color="#94A3B8", fontfamily="sans-serif")

    def draw_box(x, y, w, h, bg_col, edge_col, step_label, title, lines, title_col=None, text_size=9.2):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                             facecolor=bg_col, edgecolor=edge_col, linewidth=1.6)
        ax.add_patch(box)
        
        # Header strip inside box
        t_col = title_col if title_col else edge_col
        ax.text(x + 1.5, y + h - 0.75, step_label,
                ha="left", va="top", fontsize=9.5, fontweight="bold", color=c_stage_num)
        ax.text(x + w / 2, y + h - 0.75, title,
                ha="center", va="top", fontsize=11, fontweight="bold", color=t_col)
        
        # Horizontal subtle divider line
        ax.plot([x + 1.2, x + w - 1.2], [y + h - 1.2, y + h - 1.2], color=edge_col, linewidth=0.7, alpha=0.5)
        
        # Text lines
        n_lines = len(lines)
        if n_lines > 0:
            avail_h = h - 1.5
            spacing = avail_h / (n_lines + 0.3)
            for i, line in enumerate(lines):
                ax.text(x + w / 2, y + h - 1.5 - (i * spacing), line,
                        ha="center", va="top", fontsize=text_size, color=c_text_body)

    def draw_arrow(x1, y1, x2, y2, color=c_arrow):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle="-|>", mutation_scale=13,
                                color=color, linewidth=2.0, linestyle="solid")
        ax.add_patch(arrow)

    # =========================================================================
    # STAGE 1: LOAD IMAGE DATASET (y: 87.0, h=5.0)
    # =========================================================================
    draw_box(10, 87.0, 80, 5.0, c_box_bg, c_box_edge,
             "STAGE 1", "Load Image Dataset",
             ["Ingest Ground Surveillance Camera Feeds (HPWREN Network, Static Towers, Drones)",
              "Ingest NASA Earthdata Satellite Imagery (Terra/Aqua MODIS & Suomi-NPP/NOAA VIIRS Active Fire)",
              "Multi-Class Targets: Fire, Smoke, Environmental Distractors (Clouds, Fog, Dust, Haze), Clear Backgrounds"],
             c_text_title)

    draw_arrow(50, 87.0, 50, 84.8)

    # =========================================================================
    # STAGE 2: DATA PREPROCESSING (y: 79.8, h=5.0)
    # =========================================================================
    draw_box(10, 79.8, 80, 5.0, c_box_bg, c_box_edge,
             "STAGE 2", "Data Preprocessing",
             ["Spatial Standardization: Resizing Frames to Backbone Geometries (224 x 224 / 256 x 256)",
              "Color Space Conversion: BGR to RGB, LAB Luminance Extraction, Multi-Spectral Channel Stacking",
              "Pixel Normalization: Zero Mean, Unit Variance Standardization via Domain Dataset Parameters"],
             c_text_title)

    draw_arrow(50, 79.8, 50, 77.6)

    # =========================================================================
    # STAGE 3: DATA LEAKAGE CHECKS (y: 71.8, h=5.8)
    # =========================================================================
    draw_box(10, 71.8, 80, 5.8, c_leak_bg, c_leak_edge,
             "STAGE 3", "Data Leakage Checks (Rigorous Verification Audits)",
             ["Target Leakage Audit: Confirm Spatial Frames and Metadata Contain Zero Ground Truth Proxies or Labels",
              "Duplicate Leakage Audit: Perceptual Hashing (pHash) and MD5 Checksums Prevent Identical Images Across Splits",
              "Suspicious Features Audit: Detect and Eliminate Timestamp Watermarks, Sensor Borders, and Zero-Variance Channels"],
             c_leak_edge)

    draw_arrow(50, 71.8, 50, 69.6)

    # =========================================================================
    # STAGE 4: CLAHE (y: 64.6, h=5.0)
    # =========================================================================
    draw_box(10, 64.6, 80, 5.0, c_box_bg, c_box_edge,
             "STAGE 4", "Contrast Limited Adaptive Histogram Equalization (CLAHE)",
             ["Applied to Luminance Channel (L in LAB / Y in YCrCb) when Contrast Enhancement is Useful",
              "Enhances Low-Contrast, Diffuse Smoke Plumes in Mountainous, Mist-Covered, or Glare-Heavy Scenes",
              "Prevents Local Noise Amplification via Dynamic Local Clip Limits (Clip Limit = 2.0, Tile Grid = 8 x 8)"],
             c_text_title)

    draw_arrow(50, 64.6, 50, 62.4)

    # =========================================================================
    # STAGE 5: DATASET PARTITIONING PROTOCOL (y: 56.6, h=5.8)
    # =========================================================================
    draw_box(10, 56.6, 80, 5.8, c_box_bg, c_box_edge,
             "STAGE 5", "Dataset Partitioning Protocol & Cross-Validation",
             ["Option A: Stratified Train / Validation / Test Split (70% Train, 15% Validation, 15% Test)",
              "Option B: 10-Fold Stratified Cross-Validation (K=10) on Development Partition",
              "Strict Quarantine: Validation and Test Partitions are Isolated Prior to Any Data Augmentation"],
             c_text_title)

    draw_arrow(50, 56.6, 50, 54.4)

    # =========================================================================
    # STAGE 6: GAN AUGMENTATION (y: 48.6, h=5.8)
    # =========================================================================
    draw_box(10, 48.6, 80, 5.8, c_box_bg, c_box_edge,
             "STAGE 6", "GAN Augmentation on TRAINING Data Only",
             ["Deep Convolutional GAN (DCGAN) Architecture with Binary Cross-Entropy Adversarial Loss",
              "Synthesizes Realistic Minority Class Samples (Nascent Smoke Columns, Night Flame Fronts)",
              "STRICT LEAKAGE AUDIT: Validation and Test Sets Remain 100% Untouched with Zero Synthetic Ingestion"],
             c_text_title)

    draw_arrow(50, 48.6, 50, 46.4)

    # =========================================================================
    # STAGE 7: INITIAL MODEL TRAINING (5 HYBRID TECHNIQUES) (y: 30.6, h=15.8)
    # =========================================================================
    # Big Container Box
    big_box = FancyBboxPatch((4, 30.6), 92, 15.8, boxstyle="round,pad=0.4",
                             facecolor=c_hybrid_bg, edgecolor=c_hybrid_edge, linewidth=2.0)
    ax.add_patch(big_box)

    ax.text(5.5, 45.4, "STAGE 7", fontsize=10, fontweight="bold", color=c_stage_num)
    ax.text(50, 45.4, "Initial Model Training • 5 Hybrid Techniques Evaluation",
            ha="center", va="top", fontsize=13, fontweight="bold", color=c_hybrid_edge)
    ax.text(50, 44.2, "Concurrent Benchmarking of 5 Spatio-Temporal and Vision Architectures from Problem Statement 25",
            ha="center", va="top", fontsize=9.2, color="#475569")
    
    # 5 Sub-boxes for the 5 Hybrid Techniques side by side
    sub_w = 17.2
    sub_h = 10.8
    sub_y = 31.8
    xs = [5.5 + i * (sub_w + 1.25) for i in range(5)]

    hybrids = [
        ("Technique 1", "YOLO +\nEfficientNet +\nLSTM",
         ["Real-Time Bounding Box", "Localization (YOLO)", "Compound-Scaled Spatial", "Feature Maps (EffNet)", "Temporal Plume Sequence", "Modeling (LSTM)"]),
        ("Technique 2", "CNN +\nBiLSTM +\nAttention",
         ["Deep 2D Convolutional", "Feature Extraction", "Bidirectional Temporal", "Context Flow (BiLSTM)", "Dynamic Attention Weighting", "on Expanding Smoke"]),
        ("Technique 3", "Swin Transformer\n+ CNN",
         ["Hierarchical Shifted", "Window Self-Attention", "Multi-Scale Remote", "Sensing Representations", "Fused with Localized", "CNN Conv Edge Features"]),
        ("Technique 4", "YOLO +\nVision Transformer\n+ LSTM",
         ["Weakly Supervised", "Patch Localization", "Multi-Head Self-Attention", "Over Image Tiles (ViT)", "Long-Term Recurrent", "State Aggregation"]),
        ("Technique 5", "ConvLSTM +\nAttention",
         ["Preserves 2D Spatial", "Topology in Recurrence", "10-Day Multi-Spectral", "NetCDF Cube Ingestion", "Temporal Attention on", "Fire Spread Dynamics"])
    ]

    for idx, (x_pos, (tech_num, tech_name, tech_desc)) in enumerate(zip(xs, hybrids)):
        s_box = FancyBboxPatch((x_pos, sub_y), sub_w, sub_h, boxstyle="round,pad=0.25",
                               facecolor="#FFFFFF", edgecolor="#3B82F6", linewidth=1.4)
        ax.add_patch(s_box)
        
        ax.text(x_pos + sub_w / 2, sub_y + sub_h - 0.5, tech_num,
                ha="center", va="top", fontsize=8.5, fontweight="bold", color=c_stage_num)
        ax.text(x_pos + sub_w / 2, sub_y + sub_h - 1.4, tech_name,
                ha="center", va="top", fontsize=9.2, fontweight="bold", color="#1E3A8A", linespacing=1.0)
        
        # Inner line
        ax.plot([x_pos + 0.8, x_pos + sub_w - 0.8], [sub_y + sub_h - 4.4, sub_y + sub_h - 4.4],
                color="#93C5FD", linewidth=0.8)
        
        # Bullets
        for line_idx, line_txt in enumerate(tech_desc):
            ax.text(x_pos + sub_w / 2, sub_y + sub_h - 4.8 - (line_idx * 0.95), line_txt,
                    ha="center", va="top", fontsize=7.6, color="#334155")

    draw_arrow(50, 30.6, 50, 28.4)

    # =========================================================================
    # STAGE 8: OVERFITTING / UNDERFITTING DETECTION (y: 22.8, h=5.6)
    # =========================================================================
    draw_box(10, 22.8, 80, 5.6, c_box_bg, c_box_edge,
             "STAGE 8", "Overfitting / Underfitting Detection and Rectification",
             ["Learning Curve Auditing: Continuous Real-Time Tracking of Training Loss vs. Validation Loss",
              "Generalization Gap Analysis: Quantifying Divergence Between Training Accuracy and Validation Accuracy",
              "Diagnosis & Triggers: High Memorization Gap Triggers Phase 9 Structural and Stochastic Regularization"],
             c_text_title)

    draw_arrow(50, 22.8, 50, 20.6)

    # =========================================================================
    # STAGE 9: APPLY CORRECTION TECHNIQUE & RETRAIN (y: 14.8, h=5.8)
    # =========================================================================
    draw_box(10, 14.8, 80, 5.8, c_box_bg, c_box_edge,
             "STAGE 9", "Apply Correction Technique and Retrain",
             ["Stochastic Regularization: Spatial Dropout (0.2–0.3) and Dense Classifier Dropout (0.4–0.5)",
              "Activation Normalization: Batch Normalization Across Convolutional and Recurrent Stages",
              "Loss & Optimizer Tuning: L2 Weight Decay (1e-4), Label Smoothing (eps=0.1), and Cosine Annealing Learning Rate"],
             c_text_title)

    draw_arrow(50, 14.8, 50, 12.6)

    # =========================================================================
    # STAGE 10: FINAL TEST EVALUATION (y: 6.8, h=5.8)
    # =========================================================================
    draw_box(10, 6.8, 80, 5.8, c_box_bg, c_box_edge,
             "STAGE 10", "Final Test Evaluation (Pristine Held-Out Benchmark)",
             ["Evaluated Strictly on Quarantined, Unseen Test Split with Zero Information Leakage",
              "Comprehensive Evaluation Metrics: Accuracy, Precision, Recall, Macro F1-Score, and ROC-AUC",
              "Confusion Matrix Analysis: Rigorous Validation of Zero False Alarms on Cloud, Fog, and Dust Distractors"],
             c_text_title)

    draw_arrow(50, 6.8, 50, 4.6)

    # =========================================================================
    # OUTPUT: PROBLEM STATEMENT ALERT PIPELINE (y: 0.8, h=3.8)
    # =========================================================================
    draw_box(4, 0.8, 92, 3.8, c_out_bg, c_out_edge,
             "SYSTEM OUTPUT", "End-to-End Operational Decision Flow",
             ["Fire / Smoke Detection  -->  Severity Estimation (LOW, MED, HIGH, CRITICAL)  -->  Geographic Location  -->  Early-Warning Alert",
              "Real-time alert dispatch to GIS Command Center Dashboard with telemetry, fire perimeter, and affected area estimates."],
             c_out_edge, text_size=9.5)

    # Save outputs
    path1 = os.path.join("docs", "image_classification_flowchart.png")
    path2 = os.path.join("docs", "methodology_flowchart.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.savefig(path2, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    print(f"[OK] Generated: {path1}")
    print(f"[OK] Generated: {path2}")

if __name__ == "__main__":
    build_image_flowchart()
