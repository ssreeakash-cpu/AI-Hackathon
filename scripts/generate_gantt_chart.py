"""
Script: generate_gantt_chart.py
Description: Generates a publication-grade, high-resolution Gantt Chart
illustrating the complete 12-week development lifecycle of the Unified Multimodal
Early-Warning Wildfire Detection Pipeline.
Outputs:
  - docs/project_gantt_chart.png
  - docs/methodology_gantt_chart.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def build_gantt_chart():
    os.makedirs("docs", exist_ok=True)

    # Define tasks, start weeks, durations, and category
    tasks = [
        # Phase 1: Data Ingestion & Integrity
        {"task": "1. Multi-Modal Telemetry & Imagery Ingestion", "start": 0.0, "duration": 2.0, "phase": "Data Ingestion & Integrity", "color": "#1f77b4"},
        {"task": "2. Preprocessing & Fire Radiative Power Binning", "start": 1.0, "duration": 1.8, "phase": "Data Ingestion & Integrity", "color": "#1f77b4"},
        {"task": "3. Data Leakage Audits (Target, Duplicate, Features)", "start": 1.5, "duration": 1.5, "phase": "Data Ingestion & Integrity", "color": "#1f77b4"},
        
        # Phase 2: Feature Transformation & Stratification
        {"task": "4. Contrast Limited Adaptive Histogram Eq. (CLAHE)", "start": 2.2, "duration": 1.5, "phase": "Transformation & Balancing", "color": "#009688"},
        {"task": "5. Stratified Partitioning (65/15/20) & 10-Fold CV", "start": 3.0, "duration": 1.5, "phase": "Transformation & Balancing", "color": "#009688"},
        {"task": "6. Training-Only Resampling: DCGAN & SMOTE", "start": 3.8, "duration": 2.2, "phase": "Transformation & Balancing", "color": "#009688"},
        
        # Phase 3: Model Architecture Exploration
        {"task": "7. Initial Image Models (5 Hybrid Architectures)", "start": 4.5, "duration": 2.5, "phase": "Architecture Exploration", "color": "#3f51b5"},
        {"task": "8. Tabular Baseline Evaluation (Random Forest)", "start": 5.0, "duration": 1.8, "phase": "Architecture Exploration", "color": "#3f51b5"},
        {"task": "9. Cross-Validation Benchmarking & Top-3 Selection", "start": 6.0, "duration": 1.8, "phase": "Architecture Exploration", "color": "#3f51b5"},
        
        # Phase 4: Cascaded Architecture & Cyclical Feedback
        {"task": "10. Cascaded Top-3 Pipeline (Model 1 -> 2 -> 3)", "start": 7.0, "duration": 2.0, "phase": "Cascaded System Engineering", "color": "#673ab7"},
        {"task": "11. Cyclical Feedback Loop (Model 3 -> Model 1)", "start": 7.8, "duration": 1.8, "phase": "Cascaded System Engineering", "color": "#673ab7"},
        
        # Phase 5: Overfitting Diagnostics & Retraining
        {"task": "12. Overfit / Underfit Gap & Learning Curve Audits", "start": 8.5, "duration": 1.5, "phase": "Diagnostics & Retraining", "color": "#e65100"},
        {"task": "13. Regularization & Tree Pruning (Dropout, Smoothing)", "start": 9.0, "duration": 1.8, "phase": "Diagnostics & Retraining", "color": "#e65100"},
        {"task": "14. Post-Correction Retraining & Convergence Auditing", "start": 9.5, "duration": 1.5, "phase": "Diagnostics & Retraining", "color": "#e65100"},
        
        # Phase 6: Final Evaluation & Multimodal Deployment
        {"task": "15. Quarantined Test Evaluation & Confusion Matrices", "start": 10.0, "duration": 1.5, "phase": "Evaluation & Deployment", "color": "#2e7d32"},
        {"task": "16. Multimodal Decision Fusion Engine", "start": 10.5, "duration": 1.3, "phase": "Evaluation & Deployment", "color": "#2e7d32"},
        {"task": "17. Georeferenced Early-Warning Alert Dispatch", "start": 11.0, "duration": 1.0, "phase": "Evaluation & Deployment", "color": "#2e7d32"},
    ]

    # Invert order for top-to-bottom rendering
    tasks = tasks[::-1]

    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    y_positions = np.arange(len(tasks))
    bar_height = 0.55

    # Track unique phases for legend
    phases_tracked = {}

    for i, t in enumerate(tasks):
        start = t["start"]
        dur = t["duration"]
        col = t["color"]
        phase = t["phase"]

        # Draw main task bar with subtle rounded edge effect
        bar = ax.barh(
            y_positions[i], dur, left=start, height=bar_height,
            color=col, edgecolor='#2c3e50', linewidth=1.0, alpha=0.90, zorder=3
        )
        
        # Add text label inside or adjacent to bar
        end = start + dur
        label_text = f" W{start:.1f} - W{end:.1f} ({dur:.1f}w)"
        ax.text(
            end + 0.1, y_positions[i], label_text,
            va='center', ha='left', fontsize=8.5, fontweight='bold', color='#333333', zorder=4
        )

        if phase not in phases_tracked:
            phases_tracked[phase] = col

    # Configure axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels([t["task"] for t in tasks], fontsize=10, fontweight='bold', color='#2c3e50')
    ax.set_xlabel("Project Timeline (Weeks)", fontsize=12, fontweight='bold', color='#2c3e50', labelpad=10)
    ax.set_xlim(0, 13)
    ax.set_xticks(np.arange(0, 14, 1))
    ax.set_xticklabels([f"Week {w}" for w in range(0, 14)], fontsize=9, fontweight='bold', color='#2c3e50')

    # Add vertical grid lines for weeks
    ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7', zorder=1)
    ax.yaxis.grid(False)

    # Subtle alternating row shading
    for i in range(len(tasks)):
        if i % 2 == 0:
            ax.axhspan(i - 0.4, i + 0.4, color='#f0f4f8', alpha=0.5, zorder=0)

    # Title & Subtitle
    plt.suptitle("Unified Multimodal Wildfire Pipeline Engineering Schedule (Gantt Chart)",
                 fontsize=15, fontweight='bold', color='#1a252f', y=0.96)
    plt.title("Sequential Execution Roadmap: Data Auditing, Stratification, Top-3 Cascading, and Multimodal Alert Dispatch",
              fontsize=10.5, color='#555555', pad=14)

    # Create custom legend for phases
    legend_patches = [
        patches.Patch(facecolor=color, edgecolor='#2c3e50', label=phase, linewidth=0.8)
        for phase, color in phases_tracked.items()
    ]
    ax.legend(
        handles=legend_patches[::-1], loc='lower right', bbox_to_anchor=(1.0, 0.02),
        frameon=True, facecolor='#ffffff', edgecolor='#bdc3c7', fontsize=9,
        title="Engineering Phases", title_fontsize=10
    )

    # Spines styling
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#bdc3c7')
        ax.spines[spine].set_linewidth(1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out1 = os.path.join("docs", "project_gantt_chart.png")
    out2 = os.path.join("docs", "methodology_gantt_chart.png")
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Gantt chart to: {out1} and {out2}")

if __name__ == "__main__":
    build_gantt_chart()
