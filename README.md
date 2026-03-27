# Spatial Telemetry & Gradient Boosting Fall Detection System

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

A continuous monitoring computer vision pipeline designed to detect severe human fall events. This system maps raw video pixel data into a high-dimensional spatial manifold, evaluates biomechanical anomalies using a LightGBM decision tree ensemble, and dispatches real-time automated telephony alerts via the Twilio REST API.

## Table of Contents

- [Architectural Overview](#architectural-overview)
- [Directory Structure](#directory-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [System Execution Phases](#system-execution-phases)
- [Data Visualization Context](#data-visualization-context)
- [Future Architecture Roadmap (SORT)](#future-architecture-roadmap-sort)
- [Requirements](#requirementstxt)

## Architectural Overview

The system strictly separates data engineering, model training, and real-time inference to prevent namespace collisions and memory leaks during live execution.

1. **Computer Vision Frontend:** Google MediaPipe Tasks API maps video frames into an $\mathbb{R}^{132}$ spatial coordinate space (33 anatomical landmarks $\times$ 4 vectors: $X, Y, Z, \text{Visibility}$).
2. **Machine Learning Engine:** Scikit-Learn maintains stateful preprocessing (`MinMaxScaler`), while LightGBM optimizes the binary classification decision boundary.
3. **Telephony Integration:** Twilio bridges the local Python execution environment to the global SIP/PSTN network to execute physical phone calls.
4. **Temporal Smoothing:** A double-ended queue calculates moving averages to filter out high-frequency noise and sudden probabilistic spikes.

## Directory Structure

```text
fall-detection-system/
│
├── data/
│   ├── raw_videos/
│   │   ├── Falls/               # Positive class video dataset
│   │   └── ADL/                 # Negative control dataset (Activities of Daily Living)
│   └── processed/
│       └── extracted_features.csv # Serialized spatial coordinate matrices
│
├── models/
│   ├── lightgbm_fall_model.pkl  # Serialized gradient boosting trees
│   └── feature_scaler.pkl       # Serialized normalization boundaries
│
├── plots/                       # Model evaluation artifacts
│
└── scripts/
    ├── 01_extract_features.py   # Maps .mp4 files to continuous numerical matrices
    ├── 02_train_model.py        # Optimizes decision boundaries and generates plots
    └── 03_realtime_inference.py # Live video processing and Twilio state machine
```

## Prerequisites & Installation

This pipeline requires specific compiled C++ binaries for MediaPipe and LightGBM. It is highly recommended to run this within an isolated Conda environment.

**1. Create and activate the environment:**

```bash
conda create -n cv-pipeline python=3.10
conda activate cv-pipeline
```

**2. Install dependencies:**

Either install directly via pip or use the `requirements.txt` file.

```bash
pip install opencv-python mediapipe==0.10.21 pandas numpy scikit-learn lightgbm twilio matplotlib seaborn joblib
```

**3. Configure Telephony Credentials:**

The inference engine requires active Twilio API keys. Export these to your environment before execution:

```bash
export TWILIO_ACCOUNT_SID="your_account_sid"
export TWILIO_AUTH_TOKEN="your_auth_token"
```

## System Execution Phases

### Phase 1: Feature Extraction (`01_extract_features.py`)

Iterates through the positive (`Falls`) and negative (`ADL`) control directories. It uses the MediaPipe neural network to extract the spatial coordinates of the human skeleton frame-by-frame, structuring the data into a dense CSV matrix.

### Phase 2: Training & Serialization (`02_train_model.py`)

Loads the spatial matrices into memory. It fits a `MinMaxScaler` strictly to the training distribution to map the $X \in \mathbb{R}^{N \times 132}$ feature space to a bounded $[0, 1]^{132}$ interval. It trains the LightGBM classifier to minimize log loss, evaluates the holdout set, and serializes the stateful artifacts using `joblib`.

### Phase 3: Real-Time Inference (`03_realtime_inference.py`)

Loads the serialized `.pkl` files and executes a forward pass on a live video feed or pre-recorded `.mp4`.

- **Noise Suppression:** Utilizes a `deque` of size 10 to require a sustained, multi-frame physical collapse before breaching the conditional probability threshold.
- **Alert State Machine:** Incorporates a 150-frame temporal lockout (cooldown) after an alert is dispatched to prevent Twilio API spam during prolonged events.

## Data Visualization Context

Executing Phase 2 generates three critical diagnostic artifacts in the `/plots/` directory to verify mathematical integrity.

### 1. Top 20 Biomechanical Feature Importances (`feature_importance.png`)

Maps the internal Gini impurity reduction logic of the decision trees. The $\mathbb{R}^{132}$ feature space corresponds to 33 specific landmarks. Any coordinate vector $v_i$ can be translated back to human anatomy using modulo arithmetic: $\lfloor i/4 \rfloor$ yields the joint ID, and $i \pmod{4}$ yields the spatial axis $(X, Y, Z, V)$.

The model heavily weights vectors mapping exactly to the Y-axis (vertical displacement) of the hips.

> This verifies the algorithm reverse-engineered actual gravitational physics, anchoring its decision boundary on the rapid vertical drop of the human body's center of mass.

### 2. Classification Confusion Matrix (`confusion_matrix.png`)

Maps the Cartesian product of true labels versus predicted labels on the 20% holdout test set. Visualizes the model's recall rate, confirming False Negatives (missed falls) are mathematically isolated and minimized.

### 3. Receiver Operating Characteristic (`roc_curve.png`)

Plots the True Positive Rate against the False Positive Rate across all possible classification probability thresholds $\tau \in [0, 1]$. The Area Under the Curve (AUC) demonstrates the raw discriminative capacity of the model.

## Future Architecture Roadmap (SORT)

The current architecture utilizes a single-topology estimator. To monitor multiple individuals simultaneously, the pipeline will be upgraded to a Detection-Tracking-Estimation cascade.

Implementing **Simple Online and Realtime Tracking (SORT)** requires introducing a bounding box detector (YOLOv8) upstream of the MediaPipe extractor, combined with:

- **Kinematic State Estimation (Kalman Filter):** Models the state of a bounding box as a 7-dimensional vector: $\mathbf{x} = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$. Predicts spatial coordinates based on momentum.
- **Frame-to-Frame Data Association (Hungarian Algorithm):** Computes an assignment cost matrix using the Intersection over Union (IoU) distance between the Kalman Filter's predicted boxes and YOLO's actual detected boxes, definitively assigning a persistent ID to each subject across time.

## `requirements.txt`

If deploying to a new machine, save this block as `requirements.txt` in the root directory to ensure environment parity:

```text
opencv-python>=4.8.0
mediapipe==0.10.21
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
twilio>=8.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```
