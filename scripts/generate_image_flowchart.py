"""
Script: generate_image_flowchart.py
Description: Generates a publication-grade, high-resolution flowchart for the
Image Classification Pipeline featuring:
  - Exact 10-step sequence requested
  - 5 Hybrid Techniques Evaluation
  - Top-3 Model Cascading: Model 1 -> Model 2 -> Model 3 -> Result
  - Cyclical Feedback Loop: 3 -> 1 -> 2 -> 3
  - Output: Fire/smoke detection -> severity estimation -> geographic location -> early-warning alert.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def build_image_flowchart():
    os.makedirs("docs", exist_ok=True)
    
    # Dimensions: 20 x 34 inches at 300 DPI for high definition and readability
    fig = plt.figure(figsize=(20, 34), dpi=300)
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
    c_cascade_bg = "#FAF5FF"  # Soft Purple tint
    c_cascade_edge = "#7C3AED"# Deep Purple
    c_leak_bg = "#FEF2F2"     # Soft Red tint
    c_leak_edge = "#B91C1C"   # Deep Red
    c_out_bg = "#F0FDF4"      # Soft Green tint
    c_out_edge = "#15803D"    # Deep Forest Green

    # Header Banner
    header_box = FancyBboxPatch((4, 94.6), 92, 4.4, boxstyle="round,pad=0.4",
                                facecolor="#0F172A", edgecolor="#1E293B", linewidth=2.0)
    ax.add_patch(header_box)
    
    ax.text(50, 97.4, "AI-BASED EARLY DETECTION OF FOREST FIRE AND SMOKE",
            ha="center", va="center", fontsize=19, fontweight="bold", color="#FFFFFF", fontfamily="sans-serif")
    ax.text(50, 95.7, "Problem Statement 25 • Image Classification Pipeline & Top-3 Cascaded Architecture (1 -> 2 -> 3 with 3 -> 1 Feedback)",
            ha="center", va="center", fontsize=11.5, fontweight="medium", color="#94A3B8", fontfamily="sans-serif")

    def draw_box(x, y, w, h, bg_col, edge_col, step_label, title, lines, title_col=None, text_size=9.2):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                             facecolor=bg_col, edgecolor=edge_col, linewidth=1.6)
        ax.add_patch(box)
        
        # Header strip inside box
        t_col = title_col if title_col else edge_col
        ax.text(x + 1.5, y + h - 0.70, step_label,
                ha="left", va="top", fontsize=9.2, fontweight="bold", color=c_stage_num)
        ax.text(x + w / 2, y + h - 0.70, title,
                ha="center", va="top", fontsize=10.8, fontweight="bold", color=t_col)
        
        # Horizontal subtle divider line
        ax.plot([x + 1.2, x + w - 1.2], [y + h - 1.15, y + h - 1.15], color=edge_col, linewidth=0.7, alpha=0.5)
        
        # Text lines
        n_lines = len(lines)
        if n_lines > 0:
            avail_h = h - 1.4
            spacing = avail_h / (n_lines + 0.3)
            for i, line in enumerate(lines):
                ax.text(x + w / 2, y + h - 1.4 - (i * spacing), line,
                        ha="center", va="top", fontsize=text_size, color=c_text_body)

    def draw_arrow(x1, y1, x2, y2, color=c_arrow):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle="-|>", mutation_scale=12,
                                color=color, linewidth=1.8, linestyle="solid")
        ax.add_patch(arrow)

    # =========================================================================
    # STAGE 1: LOAD IMAGE DATASET (y: 89.2, h=4.3)
    # =========================================================================
    draw_box(10, 89.2, 80, 4.3, c_box_bg, c_box_edge,
             "STAGE 1", "Load Image Dataset",
             ["Ingest Ground Surveillance Camera Feeds (HPWREN Network, Static Towers, Drones)",
              "Ingest NASA Earthdata Satellite Imagery (Terra/Aqua MODIS & Suomi-NPP/NOAA VIIRS Active Fire)",
              "Multi-Class Targets: Fire, Smoke, Environmental Distractors (Clouds, Fog, Dust, Haze), Clear Backgrounds"],
             c_text_title)

    draw_arrow(50, 89.2, 50, 87.3)

    # =========================================================================
    # STAGE 2: DATA PREPROCESSING (y: 83.0, h=4.3)
    # =========================================================================
    draw_box(10, 83.0, 80, 4.3, c_box_bg, c_box_edge,
             "STAGE 2", "Data Preprocessing",
             ["Spatial Standardization: Resizing Frames to Backbone Geometries (224 x 224 / 256 x 256)",
              "Color Space Conversion: BGR to RGB, LAB Luminance Extraction, Multi-Spectral Channel Stacking",
              "Pixel Normalization: Zero Mean, Unit Variance Standardization via Domain Dataset Parameters"],
             c_text_title)

    draw_arrow(50, 83.0, 50, 81.1)

    # =========================================================================
    # STAGE 3: DATA LEAKAGE CHECKS (y: 75.8, h=5.3)
    # =========================================================================
    draw_box(10, 75.8, 80, 5.3, c_leak_bg, c_leak_edge,
             "STAGE 3", "Data Leakage Checks (Rigorous Verification Audits)",
             ["Target Leakage Audit: Confirm Spatial Frames and Metadata Contain Zero Ground Truth Proxies or Labels",
              "Duplicate Leakage Audit: Perceptual Hashing (pHash) and MD5 Checksums Prevent Identical Images Across Splits",
              "Suspicious Features Audit: Detect and Eliminate Timestamp Watermarks, Sensor Borders, and Zero-Variance Channels"],
             c_leak_edge)

    draw_arrow(50, 75.8, 50, 73.9)

    # =========================================================================
    # STAGE 4: CLAHE (y: 68.6, h=4.3)
    # =========================================================================
    draw_box(10, 68.6, 80, 4.3, c_box_bg, c_box_edge,
             "STAGE 4", "Contrast Limited Adaptive Histogram Equalization (CLAHE)",
             ["Applied to Luminance Channel (L in LAB / Y in YCrCb) when Contrast Enhancement is Useful",
              "Enhances Low-Contrast, Diffuse Smoke Plumes in Mountainous, Mist-Covered, or Glare-Heavy Scenes",
              "Prevents Local Noise Amplification via Dynamic Local Clip Limits (Clip Limit = 2.0, Tile Grid = 8 x 8)"],
             c_text_title)

    draw_arrow(50, 68.6, 50, 66.7)

    # =========================================================================
    # STAGE 5: DATASET PARTITIONING PROTOCOL (y: 61.4, h=5.3)
    # =========================================================================
    draw_box(10, 61.4, 80, 5.3, c_box_bg, c_box_edge,
             "STAGE 5", "Dataset Partitioning Protocol & Cross-Validation",
             ["Option A: Stratified Train / Validation / Test Split (70% Train, 15% Validation, 15% Test)",
              "Option B: 10-Fold Stratified Cross-Validation (K=10) on Development Partition",
              "Strict Quarantine: Validation and Test Partitions are Isolated Prior to Any Data Augmentation"],
             c_text_title)

    draw_arrow(50, 61.4, 50, 59.5)

    # =========================================================================
    # STAGE 6: GAN AUGMENTATION (y: 54.2, h=5.3)
    # =========================================================================
    draw_box(10, 54.2, 80, 5.3, c_box_bg, c_box_edge,
             "STAGE 6", "GAN Augmentation on TRAINING Data Only",
             ["Deep Convolutional GAN (DCGAN) Architecture with Binary Cross-Entropy Adversarial Loss",
              "Synthesizes Realistic Minority Class Samples (Nascent Smoke Columns, Night Flame Fronts)",
              "STRICT LEAKAGE AUDIT: Validation and Test Sets Remain 100% Untouched with Zero Synthetic Ingestion"],
             c_text_title)

    draw_arrow(50, 54.2, 50, 52.3)

    # =========================================================================
    # STAGE 7: INITIAL MODEL TRAINING (5 HYBRID TECHNIQUES) (y: 40.5, h=11.8)
    # =========================================================================
    big_box = FancyBboxPatch((4, 40.5), 92, 11.8, boxstyle="round,pad=0.35",
                             facecolor=c_hybrid_bg, edgecolor=c_hybrid_edge, linewidth=1.8)
    ax.add_patch(big_box)

    ax.text(5.5, 51.5, "STAGE 7A", fontsize=9.2, fontweight="bold", color=c_stage_num)
    ax.text(50, 51.5, "Initial Model Training • 5 Hybrid Techniques Evaluation",
            ha="center", va="top", fontsize=11.5, fontweight="bold", color=c_hybrid_edge)
    ax.text(50, 50.4, "Screening of the 5 Candidate Spatio-Temporal Deep Learning Architectures from Problem Statement 25",
            ha="center", va="top", fontsize=8.6, color="#475569")
    
    sub_w = 17.2
    sub_h = 7.8
    sub_y = 41.3
    xs = [5.5 + i * (sub_w + 1.25) for i in range(5)]

    hybrids = [
        ("Technique 1", "YOLO +\nEfficientNet + LSTM",
         ["Bounding Box Detector", "Compound Feature Maps", "Sequence Tracking"]),
        ("Technique 2", "CNN +\nBiLSTM + Attention",
         ["Deep 2D Conv Encoder", "Bidirectional Temporal Flow", "Dynamic Smoke Attention"]),
        ("Technique 3", "Swin Transformer\n+ CNN",
         ["Shifted Window Attn", "Multi-Scale Features", "Local Edge Convolutions"]),
        ("Technique 4", "YOLO +\nViT + LSTM",
         ["Patch Localization", "Multi-Head Tile Attention", "Recurrent Aggregator"]),
        ("Technique 5", "ConvLSTM +\nAttention",
         ["Topology Preservation", "10-Day NetCDF Sequences", "Spread Dynamics Attn"])
    ]

    for idx, (x_pos, (tech_num, tech_name, tech_desc)) in enumerate(zip(xs, hybrids)):
        s_box = FancyBboxPatch((x_pos, sub_y), sub_w, sub_h, boxstyle="round,pad=0.25",
                               facecolor="#FFFFFF", edgecolor="#3B82F6", linewidth=1.3)
        ax.add_patch(s_box)
        
        ax.text(x_pos + sub_w / 2, sub_y + sub_h - 0.45, tech_num,
                ha="center", va="top", fontsize=8.2, fontweight="bold", color=c_stage_num)
        ax.text(x_pos + sub_w / 2, sub_y + sub_h - 1.35, tech_name,
                ha="center", va="top", fontsize=8.8, fontweight="bold", color="#1E3A8A", linespacing=1.0)
        
        ax.plot([x_pos + 0.8, x_pos + sub_w - 0.8], [sub_y + sub_h - 3.8, sub_y + sub_h - 3.8],
                color="#93C5FD", linewidth=0.7)
        
        for line_idx, line_txt in enumerate(tech_desc):
            ax.text(x_pos + sub_w / 2, sub_y + sub_h - 4.2 - (line_idx * 1.05), line_txt,
                    ha="center", va="top", fontsize=7.6, color="#334155")

    draw_arrow(50, 40.5, 50, 38.6)

    # =========================================================================
    # STAGE 7B: TOP-3 CASCADED ARCHITECTURE (1 -> 2 -> 3 with 3 -> 1 Feedback) (y: 28.0, h=10.6)
    # =========================================================================
    cascade_box = FancyBboxPatch((4, 28.0), 92, 10.6, boxstyle="round,pad=0.35",
                                 facecolor=c_cascade_bg, edgecolor=c_cascade_edge, linewidth=2.0)
    ax.add_patch(cascade_box)

    ax.text(5.5, 37.8, "STAGE 7B", fontsize=9.2, fontweight="bold", color="#6D28D9")
    ax.text(50, 37.8, "Top-3 Model Sequential Cascade (1 -> 2 -> 3 with 3 -> 1 Cyclical Feedback)",
            ha="center", va="top", fontsize=11.8, fontweight="bold", color=c_cascade_edge)
    ax.text(50, 36.8, "Output of Model 1 feeds as Input to Model 2; Output of Model 2 feeds as Input to Model 3; Model 3 feeds back to Model 1",
            ha="center", va="top", fontsize=8.6, color="#4C1D95")

    # 3 Cascaded Boxes
    casc_w = 26.0
    casc_h = 6.8
    casc_y = 29.0
    c_x1 = 7.0
    c_x2 = 37.0
    c_x3 = 67.0

    # Model 1 Box
    m1_box = FancyBboxPatch((c_x1, casc_y), casc_w, casc_h, boxstyle="round,pad=0.25",
                            facecolor="#FFFFFF", edgecolor="#7C3AED", linewidth=1.5)
    ax.add_patch(m1_box)
    ax.text(c_x1 + casc_w / 2, casc_y + casc_h - 0.5, "Model 1: YOLO + EfficientNet",
            ha="center", va="top", fontsize=9.2, fontweight="bold", color="#5B21B6")
    ax.text(c_x1 + casc_w / 2, casc_y + casc_h - 1.4, "[Stage 1: Spatial Feature Extractor]",
            ha="center", va="top", fontsize=7.8, color="#6D28D9")
    m1_desc = ["• Extracts multi-scale localized feature maps", "• Detects flame/smoke bounding candidates", "• Output Vector Z1 in R^128 (passed to Model 2)"]
    for i, l in enumerate(m1_desc):
        ax.text(c_x1 + 1.2, casc_y + casc_h - 2.5 - (i * 1.3), l, fontsize=7.6, color="#1E293B")

    # Forward Arrow: 1 -> 2
    fwd_arrow_12 = FancyArrowPatch((c_x1 + casc_w, casc_y + casc_h / 2), (c_x2, casc_y + casc_h / 2),
                                    arrowstyle="-|>", mutation_scale=14, color="#7C3AED", linewidth=2.2)
    ax.add_patch(fwd_arrow_12)
    ax.text((c_x1 + casc_w + c_x2) / 2, casc_y + casc_h / 2 + 0.6, "Feed Z1", ha="center", fontsize=8.0, fontweight="bold", color="#7C3AED")

    # Model 2 Box
    m2_box = FancyBboxPatch((c_x2, casc_y), casc_w, casc_h, boxstyle="round,pad=0.25",
                            facecolor="#FFFFFF", edgecolor="#7C3AED", linewidth=1.5)
    ax.add_patch(m2_box)
    ax.text(c_x2 + casc_w / 2, casc_y + casc_h - 0.5, "Model 2: Swin Transformer + CNN",
            ha="center", va="top", fontsize=9.2, fontweight="bold", color="#5B21B6")
    ax.text(c_x2 + casc_w / 2, casc_y + casc_h - 1.4, "[Stage 2: Contextual Self-Attention]",
            ha="center", va="top", fontsize=7.8, color="#6D28D9")
    m2_desc = ["• Receives Z1 from Model 1 as token input", "• Shifted-window attention separates fog/clouds", "• Output Vector Z2 in R^128 (passed to Model 3)"]
    for i, l in enumerate(m2_desc):
        ax.text(c_x2 + 1.2, casc_y + casc_h - 2.5 - (i * 1.3), l, fontsize=7.6, color="#1E293B")

    # Forward Arrow: 2 -> 3
    fwd_arrow_23 = FancyArrowPatch((c_x2 + casc_w, casc_y + casc_h / 2), (c_x3, casc_y + casc_h / 2),
                                    arrowstyle="-|>", mutation_scale=14, color="#7C3AED", linewidth=2.2)
    ax.add_patch(fwd_arrow_23)
    ax.text((c_x2 + casc_w + c_x3) / 2, casc_y + casc_h / 2 + 0.6, "Feed Z2", ha="center", fontsize=8.0, fontweight="bold", color="#7C3AED")

    # Model 3 Box
    m3_box = FancyBboxPatch((c_x3, casc_y), casc_w, casc_h, boxstyle="round,pad=0.25",
                            facecolor="#FFFFFF", edgecolor="#7C3AED", linewidth=1.5)
    ax.add_patch(m3_box)
    ax.text(c_x3 + casc_w / 2, casc_y + casc_h - 0.5, "Model 3: CNN + BiLSTM + Attention",
            ha="center", va="top", fontsize=9.2, fontweight="bold", color="#5B21B6")
    ax.text(c_x3 + casc_w / 2, casc_y + casc_h - 1.4, "[Stage 3: Sequence Modeling & Classifier]",
            ha="center", va="top", fontsize=7.8, color="#6D28D9")
    m3_desc = ["• Receives Z2 from Model 2 as sequence input", "• Dynamic temporal attention over smoke motion", "• Generates Final Classification & Feedback"]
    for i, l in enumerate(m3_desc):
        ax.text(c_x3 + 1.2, casc_y + casc_h - 2.5 - (i * 1.3), l, fontsize=7.6, color="#1E293B")

    # Cyclical Feedback Arrow: 3 -> 1 (curving below the boxes)
    back_arrow = FancyArrowPatch((c_x3 + casc_w / 2, casc_y), (c_x1 + casc_w / 2, casc_y),
                                 connectionstyle="arc3,rad=-0.16", arrowstyle="-|>",
                                 mutation_scale=13, color="#DC2626", linewidth=2.0, linestyle="dashed")
    ax.add_patch(back_arrow)
    ax.text(50, casc_y - 0.75, "Cyclical Feedback (3 -> 1): High-level temporal attention prior modulates Model 1 spatial sensitivity",
            ha="center", va="top", fontsize=8.2, fontweight="bold", color="#DC2626")

    draw_arrow(50, 28.0, 50, 26.1)

    # =========================================================================
    # STAGE 8: OVERFITTING / UNDERFITTING DETECTION (y: 20.8, h=5.3)
    # =========================================================================
    draw_box(10, 20.8, 80, 5.3, c_box_bg, c_box_edge,
             "STAGE 8", "Overfitting / Underfitting Detection and Rectification",
             ["Learning Curve Auditing: Continuous Real-Time Tracking of Training Loss vs. Validation Loss",
              "Generalization Gap Analysis: Quantifying Divergence Between Training Accuracy and Validation Accuracy",
              "Diagnosis & Triggers: High Memorization Gap Triggers Phase 9 Structural and Stochastic Regularization"],
             c_text_title)

    draw_arrow(50, 20.8, 50, 18.9)

    # =========================================================================
    # STAGE 9: APPLY CORRECTION TECHNIQUE & RETRAIN (y: 13.6, h=5.3)
    # =========================================================================
    draw_box(10, 13.6, 80, 5.3, c_box_bg, c_box_edge,
             "STAGE 9", "Apply Correction Technique and Retrain",
             ["Stochastic Regularization: Spatial Dropout (0.2–0.3) and Dense Classifier Dropout (0.4–0.5)",
              "Activation Normalization: Batch Normalization Across Convolutional and Recurrent Stages",
              "Loss & Optimizer Tuning: L2 Weight Decay (1e-4), Label Smoothing (eps=0.1), and Cosine Annealing Learning Rate"],
             c_text_title)

    draw_arrow(50, 13.6, 50, 11.7)

    # =========================================================================
    # STAGE 10: FINAL TEST EVALUATION (y: 6.4, h=5.3)
    # =========================================================================
    draw_box(10, 6.4, 80, 5.3, c_box_bg, c_box_edge,
             "STAGE 10", "Final Test Evaluation (Pristine Held-Out Benchmark)",
             ["Evaluated Strictly on Quarantined, Unseen Test Split with Zero Information Leakage",
              "Comprehensive Evaluation Metrics: Accuracy, Precision, Recall, Macro F1-Score, and ROC-AUC",
              "Confusion Matrix Analysis: Rigorous Validation of Zero False Alarms on Cloud, Fog, and Dust Distractors"],
             c_text_title)

    draw_arrow(50, 6.4, 50, 4.5)

    # =========================================================================
    # OUTPUT: PROBLEM STATEMENT ALERT PIPELINE (y: 0.6, h=3.9)
    # =========================================================================
    draw_box(4, 0.6, 92, 3.9, c_out_bg, c_out_edge,
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
