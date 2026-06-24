# ===============================
# IMPORTS
# ===============================
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import StandardScaler


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="CardioVision – ECG Arrhythmia Detection",
    page_icon="🫀",
    layout="wide"
)


# ===============================
# LABELS
# ===============================
LABELS = {
    0: "Normal Beat (N)",
    1: "Supraventricular Ectopic Beat (SVEB)",
    2: "Ventricular Ectopic Beat (PVC)",
    3: "Fusion Beat (F)",
    4: "Unknown Beat (Q)"
}

# Support string labels in CSV
LABEL_MAP_STR = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4
}


# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path, compile=False)

    # warm-up with correct input length
    dummy = np.zeros((1, model.input_shape[1], 1), dtype=np.float32)
    model.predict(dummy, verbose=0)

    return model


# ===============================
# NORMALIZATION
# ===============================
def normalize(signal):
    scaler = StandardScaler()
    signal = scaler.fit_transform(signal.reshape(-1, 1))
    return signal.flatten()


# ===============================
# RESIZE ECG TO MODEL INPUT
# ===============================
def resize_signal(signal, target_len):
    return np.interp(
        np.linspace(0, len(signal) - 1, target_len),
        np.arange(len(signal)),
        signal
    )


# ===============================
# GRADIENT-BASED CAM (CNN-LSTM SAFE)
# ===============================
def grad_cam_input(model, signal):
    """
    Stable gradient-based CAM on input signal.
    Works reliably for CNN-LSTM architectures.
    """

    x = tf.convert_to_tensor(signal.reshape(1, -1, 1), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, x)[0].numpy().flatten()

    cam = np.abs(grads)

    # smooth CAM for ECG
    cam = np.convolve(cam, np.ones(7) / 7, mode="same")

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam, preds.numpy()[0]


# ===============================
# PLOT FUNCTION
# ===============================
def plot_ecg_with_cam(signal, cam, pred_class, confidence, true_label=None):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(signal, color="black", linewidth=1.4)

    ax.imshow(
        cam[np.newaxis, :],
        cmap="jet",
        alpha=0.45,
        aspect="auto",
        extent=[0, len(signal), signal.min(), signal.max()]
    )

    title = f"Prediction: {LABELS[pred_class]} | Confidence: {confidence*100:.1f}%"
    if true_label is not None:
        title += f" | True: {LABELS[true_label]}"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    return fig


# ===============================
# UI
# ===============================
st.title("🫀 CardioVision – ECG Arrhythmia Detection with Explainability")

model_path = st.sidebar.text_input(
    "Model Path",
    value=r"C:/Users/neham/Downloads/arryhthmia.serious/arrhythmia_cnn_lstm_model.h5"
)

# Load model
try:
    model = load_model(model_path)
    st.sidebar.success("Model Loaded Successfully ✔")
except Exception as e:
    st.sidebar.error("Model Load Failed ❌")
    st.error(e)
    st.stop()


# ===============================
# FILE UPLOAD
# ===============================
uploaded = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])

if uploaded is None:
    st.info("👈 Upload ECG CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded, header=None)

if df.shape[1] < 187:
    st.error("CSV must contain ECG samples + label column")
    st.stop()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

st.success(f"Loaded {len(df)} ECG records ✔")


# ===============================
# RECORD SELECTION
# ===============================
idx = st.number_input("Select ECG Record", 0, len(df) - 1, 0)

# preprocess signal
signal = normalize(X[idx])
signal = resize_signal(signal, model.input_shape[1])

# parse true label safely
true_label = None
raw_label = y[idx]

if isinstance(raw_label, str):
    true_label = LABEL_MAP_STR.get(raw_label.strip(), None)
else:
    try:
        true_label = int(raw_label)
    except:
        true_label = None


# ===============================
# PREDICTION + CAM
# ===============================
cam, preds = grad_cam_input(model, signal)
pred_class = np.argmax(preds)
confidence = preds[pred_class]


# ===============================
# DISPLAY
# ===============================
fig = plot_ecg_with_cam(signal, cam, pred_class, confidence, true_label)
st.pyplot(fig)


# ===============================
# DOWNLOAD
# ===============================
buf = BytesIO()
fig.savefig(buf, format="png", dpi=300)
buf.seek(0)

st.download_button(
    "💾 Download Visualization",
    buf,
    "ecg_explainability.png",
    "image/png"
)


