import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ==========================================
# Configuration
# ==========================================
PROJECT_ROOT = "/Users/ishanibakshi/fall-detection-system"
INPUT_CSV = os.path.join(PROJECT_ROOT, "data/processed/extracted_features.csv")
MODEL_OUT_PATH = os.path.join(PROJECT_ROOT, "models/lightgbm_fall_model.pkl")
SCALER_OUT_PATH = os.path.join(PROJECT_ROOT, "models/feature_scaler.pkl")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Guarantee the output directories exist before execution
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

def generate_visualizations(model, X_test, y_test, y_pred, y_prob):
    print(f"Generating diagnostic visualizations in {PLOTS_DIR}...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["ADL (0)", "Fall (1)"], 
                yticklabels=["ADL (0)", "Fall (1)"])
    plt.title('Classification Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=300)
    plt.close()

    # 3. Feature Importance (Top 20 Features)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Biomechanical Feature Importances (Gini Reduction)")
    plt.bar(range(20), importances[indices], align="center", color='#2ca02c')
    plt.xticks(range(20), [f"v{i}" for i in indices], rotation=45)
    plt.xlim([-1, 20])
    plt.xlabel('MediaPipe Coordinate Vector Index')
    plt.ylabel('Relative Importance')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print("Visualizations successfully saved.")

def train_and_serialize():
    print(f"Loading feature matrix from {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError("Extracted features CSV not found. Run 01_extract_features.py first.")

    df = pd.read_csv(INPUT_CSV)
    
    feature_cols = [f"v{i}" for i in range(132)]
    X = df[feature_cols]
    y = df['anomaly']

    print(f"Dataset dimensionality: {X.shape[0]} samples, {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Normalizing spatial coordinates...")
    scaler = MinMaxScaler()
    
    # Recast the scaled NumPy arrays back into DataFrames to retain feature headers
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    print("Training LightGBM spatial classifier...")
    model = LGBMClassifier(
        learning_rate=0.0379, 
        max_depth=9, 
        n_estimators=121,
        verbose=-1
    )
    model.fit(X_train_scaled, y_train)

    print("\nEvaluating model performance on holdout test set:")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] 
    
    print(classification_report(y_test, y_pred, target_names=["ADL (0)", "Fall (1)"]))

    generate_visualizations(model, X_test, y_test, y_pred, y_prob)

    print("Serializing model and preprocessing artifacts...")
    joblib.dump(model, MODEL_OUT_PATH)
    joblib.dump(scaler, SCALER_OUT_PATH)
    
    print(f"Deployment artifacts saved to:\n - {MODEL_OUT_PATH}\n - {SCALER_OUT_PATH}")

if __name__ == "__main__":
    train_and_serialize()