import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time

from PIL import Image
from tensorflow.keras.utils import img_to_array

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Glasses Classification",
    page_icon="🕶️",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model_glasses_mobilenetv2.keras",
        compile=False
    )   
    return model

model = load_model()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #f8fafc, #e2e8f0);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 48px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
}

.subtitle {
    font-size: 18px;
    color: #475569;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-box {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    padding: 25px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-top: 20px;
}

.result-label {
    font-size: 28px;
    font-weight: bold;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #64748b;
    font-size: 14px;
}

.small-text {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title">Glasses Classification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deteksi penggunaan kacamata menggunakan Deep Learning berbasis TensorFlow dan MobileNetV2.</div>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Pengaturan")

confidence_toggle = st.sidebar.checkbox(
    "Tampilkan Confidence Score",
    value=True
)

show_history = st.sidebar.checkbox(
    "Tampilkan History Prediksi",
    value=True
)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["Prediksi", "Tentang Model"])

# =========================
# TAB 1
# =========================
with tab1:

    col1, col2 = st.columns([1,1])

    # =========================
    # LEFT SIDE
    # =========================
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Input Gambar")

        input_method = st.radio(
            "Pilih sumber gambar",
            ["Upload Gambar", "Gunakan Kamera"]
        )

        uploaded_file = None

        if input_method == "Upload Gambar":

            uploaded_file = st.file_uploader(
                "Upload gambar JPG / PNG",
                type=["jpg", "jpeg", "png"]
            )

        else:

            uploaded_file = st.camera_input(
                "Ambil gambar menggunakan kamera"
            )

        if uploaded_file:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Preview Gambar",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # RIGHT SIDE
    # =========================
    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Hasil Prediksi")

        if uploaded_file:

            with st.spinner("Sedang memproses gambar..."):

                time.sleep(1)

                # =========================
                # PREPROCESSING
                # =========================
                img = image.resize((128, 128))

                img_array = img_to_array(img)
                img_array = img_array / 255.0

                img_array = np.expand_dims(img_array, axis=0)

                # =========================
                # PREDICTION
                # =========================
                prediction = model.predict(img_array)[0][0]

                # =========================
                # LABEL
                # =========================
                if prediction > 0.5:

                    label = "Tidak Menggunakan Kacamata"
                    confidence = float(prediction)

                else:

                    label = "Menggunakan Kacamata"
                    confidence = float(1 - prediction)

                # =========================
                # RESULT DISPLAY
                # =========================
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            {label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # =========================
                # METRICS
                # =========================
                col_metric1, col_metric2 = st.columns(2)

                with col_metric1:
                    st.metric(
                        "Confidence",
                        f"{confidence*100:.2f}%"
                    )

                with col_metric2:
                    st.metric(
                        "Prediction Score",
                        f"{prediction:.4f}"
                    )

                # =========================
                # PROGRESS BAR
                # =========================
                if confidence_toggle:

                    st.write("Confidence Level")

                    st.progress(confidence)

                # =========================
                # SAVE HISTORY
                # =========================
                history_data = pd.DataFrame({
                    "Prediction": [label],
                    "Confidence": [round(confidence * 100, 2)]
                })

                if os.path.exists("history.csv"):

                    history_data.to_csv(
                        "history.csv",
                        mode='a',
                        header=False,
                        index=False
                    )

                else:

                    history_data.to_csv(
                        "history.csv",
                        index=False
                    )

        else:

            st.info("Silakan upload gambar atau gunakan kamera.")

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # HISTORY
    # =========================
    if show_history:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("History Prediksi")

        if os.path.exists("history.csv"):

            history_df = pd.read_csv("history.csv")

            st.dataframe(
                history_df.tail(10),
                use_container_width=True
            )

        else:

            st.write("Belum ada history prediksi.")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TAB 2
# =========================
with tab2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Tentang Model")

    st.write("""
    Model Deep Learning ini menggunakan arsitektur MobileNetV2
    untuk melakukan klasifikasi gambar apakah seseorang
    menggunakan kacamata atau tidak.

    ### Fitur Aplikasi
    - Upload gambar JPG/PNG
    - Prediksi real-time
    - Webcam capture
    - Confidence score
    - Riwayat prediksi
    - Tampilan interaktif modern

    ### Teknologi
    - Streamlit
    - TensorFlow
    - MobileNetV2
    - NumPy
    - Pandas
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown(
    '<div class="footer">Created with Streamlit • TensorFlow • MobileNetV2</div>',
    unsafe_allow_html=True
)