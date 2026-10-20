"""
Master Pipeline Runner: Executes both Image and Tabular ML Workflows
Sequences:
1. Image Classification: Load -> Preprocess -> Leakage Checks -> CLAHE -> Splits/CV -> GAN Augmentation (Train only) -> Initial Train -> Overfit Diagnosis -> Correction -> Final Evaluation
2. CSV Classification: Load -> Preprocess -> Leakage Checks -> Splits/CV -> SMOTE (Train only) -> Initial Train -> Overfit Diagnosis -> Correction -> Final Evaluation
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipelines.image_classification_pipeline import run_pipeline as run_image_pipeline
from pipelines.tabular_classification_pipeline import run_pipeline as run_tabular_pipeline

def main():
    print("=" * 80)
    print("   AI WILDFIRE RIGOROUS ML PIPELINES (IMAGE + CSV)")
    print("=" * 80)
    
    print("\n>>> LAUNCHING PIPELINE 1: IMAGE CLASSIFICATION (CLAHE + GAN + OVERFIT RECTIFICATION)")
    img_results = run_image_pipeline()
    
    print("\n>>> LAUNCHING PIPELINE 2: CSV / TABULAR CLASSIFICATION (SMOTE + REGULARIZATION)")
    tab_results = run_tabular_pipeline()
    
    print("\n" + "=" * 80)
    print("   SUMMARY OF BOTH PIPELINES EXECUTED SUCCESSFULLY")
    print("=" * 80)
    print(f"1. Image Pipeline Final Test Accuracy:   {img_results['test_accuracy']:.2%}")
    print(f"2. Tabular Pipeline Final Test Accuracy: {tab_results['test_accuracy']:.2%}")
    print(f"   Tabular Balanced Accuracy (SMOTE):    {tab_results['test_balanced_accuracy']:.2%}")
    print(f"   Tabular Test ROC-AUC:                 {tab_results['test_roc_auc']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
