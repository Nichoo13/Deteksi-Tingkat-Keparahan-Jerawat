import streamlit as st
import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from PIL import Image

# 1. Konfigurasi Halaman (Biar rapi)
st.set_page_config(page_title="Deteksi Jerawat", page_icon="🩺", layout="centered")

st.title("🩺 Deteksi Tingkat Keparahan Jerawat pada Kulit Wajah")
st.markdown("Unggah foto wajahmu, dan biarkan sistem menganalisis tingkat keparahan jerawat dalam hitungan detik!")
st.write("---")

# 2. Muat Model (Pakai st.cache_resource agar web tidak lemot me-load ulang model terus menerus)
@st.cache_resource
def load_models():
    mata_yolo = YOLO('runs/detect/train/weights/best.pt') # Ganti path kalau beda
    otak_rf = joblib.load('rf_jerawat_model.pkl')
    alat_scaler = joblib.load('scaler_jerawat.pkl')
    return mata_yolo, otak_rf, alat_scaler

mata_yolo, otak_rf, alat_scaler = load_models()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# 3. Fitur Upload Gambar
file_upload = st.file_uploader("Pilih foto wajah (Format JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if file_upload is not None:
    # Ubah file yang diupload menjadi format yang bisa dibaca OpenCV
    image_pil = Image.open(file_upload).convert('RGB')
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    st.write("Menganalisis wajah...")
    progress_bar = st.progress(0)

    # --- PENCUCIAN VIRTUAL ---
    progress_bar.progress(30)
    img_blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    img_YCrCb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(img_YCrCb, lower_skin, upper_skin)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    img_final = cv2.bitwise_and(img_clahe, img_clahe, mask=mask)

    h_img, w_img, _ = img_final.shape
    luas_kanvas = h_img * w_img

    # --- DETEKSI YOLO ---
    progress_bar.progress(60)
    hasil_yolo = mata_yolo.predict(source=img_final, conf=0.25, verbose=False)
    kotak_yolo = hasil_yolo[0].boxes.xywh.cpu().numpy()
    skor_yolo = hasil_yolo[0].boxes.conf.cpu().numpy()

    jml_jerawat = len(kotak_yolo)
    luas_total = 0.0
    luas_terbesar = 0.0

    for kotak in kotak_yolo:
        area_px = kotak[2] * kotak[3]
        luas_total += area_px
        if area_px > luas_terbesar:
            luas_terbesar = area_px

    luas_terbesar_norm = luas_terbesar / luas_kanvas if luas_kanvas > 0 else 0
    luas_total_norm = luas_total / luas_kanvas if luas_kanvas > 0 else 0
    rata_conf = float(np.mean(skor_yolo)) if jml_jerawat > 0 else 0.0

    # --- VONIS RANDOM FOREST ---
    progress_bar.progress(90)
    fitur = np.array([[jml_jerawat, luas_terbesar_norm, luas_total_norm, rata_conf]])
    fitur_scaled = alat_scaler.transform(fitur)
    vonis = otak_rf.predict(fitur_scaled)[0]

    # Gambar kotak jerawat ke foto asli (bukan yang hitam-hitam) biar user gak takut
    img_pamer = hasil_yolo[0].plot(img=img_bgr) 
    progress_bar.progress(100)

    # 4. Tampilkan Hasil yang Cantik
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Foto Asli")
        st.image(image_pil, use_container_width=True)
    with col2:
        st.subheader("Hasil Analisis")
        st.image(cv2.cvtColor(img_pamer, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.write("---")
    st.markdown("<h3 style='text-align: center;'>Vonis AI:</h3>", unsafe_allow_html=True)

    # Bikin warna-warni tergantung keparahannya
    if vonis == "Bebas Jerawat":
        st.success(f"🌟 KONDISI KULIT: {vonis.upper()}")
    elif vonis == "Ringan":
        st.info(f"🟢 KONDISI KULIT: {vonis.upper()}")
    elif vonis == "Sedang":
        st.warning(f"🟡 KONDISI KULIT: {vonis.upper()}")
    else:
        st.error(f"🔴 KONDISI KULIT: {vonis.upper()}")

    st.write(f"**Statistik:** Sistem mendeteksi **{jml_jerawat}** area jerawat.")
