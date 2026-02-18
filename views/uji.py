import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import joblib

from skimage.feature import hog

# ==================== KONFIGURASI ==================== #
MODEL_DIR = "saved_models"
RIWAYAT_CSV = "riwayat_latih.csv"

# ==================== PREPROCESSING ==================== #
def preprocess_and_crop_image(image_path, target_size=(50, 50)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Gagal memuat gambar")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = gray[y:y+h, x:x+w]
    else:
        cropped = gray

    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

# ==================== FEATURE EXTRACTION ==================== #
def extract_hog_features(image):
    features, _ = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    return features

def extract_zoning_features(image, zones=(4, 4)):
    h, w = image.shape
    zh, zw = h // zones[0], w // zones[1]
    features = []
    for i in range(zones[0]):
        for j in range(zones[1]):
            zone = image[i*zh:(i+1)*zh, j*zw:(j+1)*zw]
            features.append(np.mean(zone))
    return np.array(features)

def extract_features(image_path, method, pca):
    img = preprocess_and_crop_image(image_path)

    if method == "Tanpa Ekstraksi":
        return img.flatten()
    if method == "HOG":
        return extract_hog_features(img)
    if method == "Zoning":
        return extract_zoning_features(img)
    if method == "PCA":
        return pca.transform(img.flatten().reshape(1, -1))[0]
    if method == "HOG + Zoning":
        return np.concatenate([extract_hog_features(img), extract_zoning_features(img)])
    if method == "HOG + PCA":
        return np.concatenate([extract_hog_features(img), pca.transform(img.flatten().reshape(1, -1))[0]])
    if method == "Zoning + PCA":
        return np.concatenate([extract_zoning_features(img), pca.transform(img.flatten().reshape(1, -1))[0]])
    if method == "HOG + Zoning + PCA":
        return np.concatenate([
            extract_hog_features(img),
            extract_zoning_features(img),
            pca.transform(img.flatten().reshape(1, -1))[0]
        ])

# ==================== UI ==================== #
def render():
    st.markdown(
        "<h1 style='text-align:center;color:#C62828'>UJI DIGIT PEMILUDIGIT</h1>",
        unsafe_allow_html=True
    )

    # ===== VALIDASI RIWAYAT =====
    if not os.path.exists(RIWAYAT_CSV):
        st.error("Belum ada model yang dilatih. Silakan latih model terlebih dahulu.")
        return

    riwayat_df = pd.read_csv(RIWAYAT_CSV)

    # ===== PATCH UNTUK CSV LAMA =====
    if "Model File" not in riwayat_df.columns:
        riwayat_df["Model File"] = riwayat_df.apply(
            lambda row: f"{row['Klasifikasi'].lower()}_{row['Ekstraksi Fitur'].replace(' ', '_')}_split{row['Test Split']}.pkl",
            axis=1
        )

    # ===== PILIH MODEL =====
    model_file = st.selectbox(
        "Pilih Model Hasil Pelatihan",
        riwayat_df["Model File"].tolist()
    )

    uploaded_file = st.file_uploader(
        "Upload gambar digit tulisan tangan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        os.makedirs("temp", exist_ok=True)
        img_path = os.path.join("temp", uploaded_file.name)

        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(img_path, caption="Gambar Uji", width=200)

        if st.button("Prediksi Digit"):
            try:
                # Ambil baris metadata model
                row = riwayat_df[riwayat_df["Model File"] == model_file].iloc[0]

                feature_method = row["Ekstraksi Fitur"]
                model_path = os.path.join(MODEL_DIR, model_file)

                scaler_path = os.path.join(
                    MODEL_DIR, f"scaler_{model_file.replace('.pkl','')}.pkl"
                )
                pca_path = os.path.join(
                    MODEL_DIR, f"pca_{model_file.replace('.pkl','')}.pkl"
                )

                # Load model pendukung
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                pca = joblib.load(pca_path)

                features = extract_features(img_path, feature_method, pca)
                features = scaler.transform([features])

                prediction = model.predict(features)[0]

                st.success(f"Hasil Prediksi Digit: **{prediction}**")

            except Exception as e:
                st.error(f"Prediksi gagal: {e}")

if __name__ == "__main__":
    render()
