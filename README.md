CardioVision – ECG Arrhythmia Detection with Explainability

CardioVision is a deep learning–based ECG arrhythmia detection system built using a CNN–LSTM architecture, deployed through an interactive Streamlit dashboard, and enhanced with explainability using gradient-based Class Activation Mapping (CAM).

This project is designed as an academic / final-year project with a strong focus on robust deployment and interpretability.

Features

CNN–LSTM model for ECG arrhythmia classification

Streamlit-based interactive dashboard

Supports MIT-BIH / PTBDB-style ECG CSV files

Handles variable ECG lengths (e.g., 187 or 216 samples)

Robust inference pipeline (no crashes on invalid inputs)

Gradient-based explainability heatmap for ECG signals

Supports both numeric and symbolic labels (0–4 or N, S, V, F, Q)

  Model Overview

Architecture: Convolutional Neural Network + LSTM

Input: Single ECG beat (1D signal)

Output Classes:

0 – Normal Beat (N)

1 – Supraventricular Ectopic Beat (SVEB)

2 – Ventricular Ectopic Beat (PVC)

3 – Fusion Beat (F)  

4 – Unknown Beat (Q)

⚠️ The trained .h5 model file is not included in this repository due to GitHub size limits.

Dashboard Preview

The dashboard allows users to:

Upload ECG CSV files

Select individual ECG records

View predicted class and confidence

Visualize ECG signal with an explainability heatmap

Download the visualization as an image

Input Data Format

CSV file

Each row represents one ECG record

Last column: label (optional)

Numeric (0–4) or

Symbolic (N, S, V, F, Q)

Remaining columns: ECG signal values

▶️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit dashboard
streamlit run dashboard_ecg.py - command

3️⃣ Provide model path

Enter the local path to your trained .h5 model in the sidebar when prompted.

🧪 Explainability Approach

Classical Grad-CAM is often unstable for CNN–LSTM architectures due to temporal dependencies introduced by LSTM layers.

To address this, CardioVision uses a gradient-based CAM on the input ECG signal, which:

Provides reliable temporal importance visualization

Highlights clinically relevant ECG regions

Works consistently at inference time

This approach is commonly adopted in ECG deep learning research.
