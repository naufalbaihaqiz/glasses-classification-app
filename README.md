# Glasses Classification System

Aplikasi web untuk mendeteksi penggunaan kacamata menggunakan Deep Learning berbasis **TensorFlow** dan **MobileNetV2**.

---

## Deskripsi

Glasses Classification System adalah aplikasi yang mampu mengklasifikasikan apakah seseorang menggunakan kacamata atau tidak dari sebuah gambar. Aplikasi ini dibangun menggunakan arsitektur transfer learning MobileNetV2 yang telah di-fine tune dengan dataset kacamata.

---

## Fitur

- 📤 Upload gambar JPG/PNG
- 📷 Prediksi real-time via kamera
- 📊 Confidence score & prediction score
- 📈 Progress bar confidence level
- 🗂️ Riwayat prediksi

---

## 🧠 Model

| Detail | Keterangan |
|--------|-----------|
| Arsitektur | MobileNetV2 (Transfer Learning) |
| Input Size | 128 x 128 x 3 |
| Output | Binary (Glasses / No Glasses) |
| Format | `.keras` |
| Versi | v2 (fine-tuned) |
| Val Accuracy | ~92.5% |

### Arsitektur Model
```
MobileNetV2 (frozen layers)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(1, sigmoid)
```
## Training

Model dilatih menggunakan Google Colab dengan dataset:
- **Train:** 500 gambar per kelas (hasil augmentasi dari 52 gambar asli)
- **Validasi:** 20 gambar per kelas
- **Epochs:** 20 (dengan EarlyStopping)
- **Fine Tuning:** 30 layer terakhir MobileNetV2

**note : model masih belum sempurna karena dataset yang sangat sedikit**
https://glasses-classification-app.streamlit.app/
