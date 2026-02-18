import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import joblib

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ======================================================
# KONFIGURASI
# ======================================================
MODEL_DIR = "saved_models"
RIWAYAT_CSV = "riwayat_latih.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# UTIL
# ======================================================
def safe_name(text: str) -> str:
    return text.lower().replace(" ", "_").replace("+", "plus")

# ======================================================
# SESSION STATE
# ======================================================
if "riwayat_latih" not in st.session_state:
    if os.path.exists(RIWAYAT_CSV):
        st.session_state.riwayat_latih = pd.read_csv(RIWAYAT_CSV).to_dict("records")
    else:
        st.session_state.riwayat_latih = []

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_and_crop_image(image_path, target_size=(50, 50)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gagal memuat gambar: {image_path}")

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

# ======================================================
# FEATURE EXTRACTION
# ======================================================
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

# ======================================================
# UI
# ======================================================
def render():
    # ==================== STYLE ==================== #
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab"] {
            background: #FFF0F0;
            border: 1px solid #F2B8B8;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            color: #C62828;
        }
        .stTabs [aria-selected="true"] {
            background: #C62828 !important;
            color: white !important;
        }
        .box {
            background: #FFF5F5;
            border-left: 6px solid #C62828;
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            margin-top: 1.2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<h1 style='text-align:center;color:#C62828'>LATIH MODEL PEMILUDIGIT</h1>",
        unsafe_allow_html=True
    )

    tab_latih, tab_riwayat = st.tabs(["Latih", "Riwayat"])

    # ==================================================
    # TAB LATIH
    # ==================================================
    with tab_latih:
        with st.container():
            dataset_path = st.text_input(
                "Path folder dataset (nama file diawali digit, contoh: 7_012.png)"
            )

            img_files = []
            if dataset_path and os.path.exists(dataset_path):
                for root, _, files in os.walk(dataset_path):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            img_files.append(os.path.join(root, f))

            if dataset_path and len(img_files) == 0:
                st.error("Dataset kosong atau path tidak valid.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            if img_files:
                labels = [int(os.path.basename(f)[0]) for f in img_files]
                if len(set(labels)) < 2:
                    st.error("Dataset harus memiliki minimal 2 kelas digit.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

                feature_method = st.selectbox(
                    "Ekstraksi Fitur",
                    [
                        "Tanpa Ekstraksi",
                        "HOG",
                        "Zoning",
                        "PCA",
                        "HOG + Zoning",
                        "HOG + PCA",
                        "Zoning + PCA",
                        "HOG + Zoning + PCA"
                    ]
                )

                classifier_name = st.selectbox("Metode Klasifikasi", ["SVM", "KNN"])
                test_split = st.selectbox("Test Split", [0.2, 0.3, 0.4])

                if st.button("Mulai Pelatihan"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        img_files, labels,
                        test_size=test_split,
                        random_state=42,
                        stratify=labels
                    )

                    train_flat = [preprocess_and_crop_image(f).flatten() for f in X_train]
                    pca = PCA(n_components=10, random_state=42)
                    pca.fit(train_flat)

                    X_train_feat = np.array([extract_features(f, feature_method, pca) for f in X_train])
                    X_test_feat = np.array([extract_features(f, feature_method, pca) for f in X_test])

                    scaler = StandardScaler()
                    X_train_feat = scaler.fit_transform(X_train_feat)
                    X_test_feat = scaler.transform(X_test_feat)

                    clf = (
                        SVC(kernel="linear", C=10, probability=True)
                        if classifier_name == "SVM"
                        else KNeighborsClassifier(n_neighbors=5)
                    )

                    clf.fit(X_train_feat, y_train)
                    acc = accuracy_score(y_test, clf.predict(X_test_feat)) * 100

                    model_tag = f"{safe_name(classifier_name)}_{safe_name(feature_method)}_split{test_split}"

                    joblib.dump(clf, f"{MODEL_DIR}/{model_tag}.pkl")
                    joblib.dump(pca, f"{MODEL_DIR}/pca_{model_tag}.pkl")
                    joblib.dump(scaler, f"{MODEL_DIR}/scaler_{model_tag}.pkl")

                    new_row = {
                        "Ekstraksi Fitur": feature_method,
                        "Klasifikasi": classifier_name,
                        "Test Split": test_split,
                        "Akurasi (%)": round(acc, 2),
                        "Model File": f"{model_tag}.pkl"
                    }

                    st.session_state.riwayat_latih.append(new_row)
                    pd.DataFrame(st.session_state.riwayat_latih).to_csv(RIWAYAT_CSV, index=False)

                    st.success(f"Akurasi Model: **{acc:.2f}%**")

            st.markdown('</div>', unsafe_allow_html=True)

    # ==================================================
    # TAB RIWAYAT
    # ==================================================
    with tab_riwayat:
        with st.container():

            if st.session_state.riwayat_latih:
                st.dataframe(pd.DataFrame(st.session_state.riwayat_latih), use_container_width=True)
            else:
                st.info("Belum ada riwayat pelatihan.")

            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    render()