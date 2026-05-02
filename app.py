import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops

# Load models
model = joblib.load("rf_model.pkl")
selector = joblib.load("selector.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Anemia Detection from Fingernail Image")

# Feature extraction
def extract_features(img):

    img = cv2.resize(img, (224,224))

    rgb_mean = np.mean(img.reshape(-1,3), axis=0)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_mean = np.mean(hsv.reshape(-1,3), axis=0)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_mean = np.mean(lab.reshape(-1,3), axis=0)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb_mean = np.mean(ycrcb.reshape(-1,3), axis=0)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]

    features = np.concatenate([
        rgb_mean, hsv_mean, lab_mean, ycrcb_mean,
        [contrast, homogeneity, energy, correlation]
    ])

    return features


# Upload image
uploaded_file = st.file_uploader("Upload fingernail image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Anemia"):

        features = extract_features(img).reshape(1,-1)

        features = selector.transform(features)
        features = scaler.transform(features)

        prob = model.predict_proba(features)[0][1]

        st.subheader("Result")

        st.progress(int(prob * 100))
        st.write(f"Anemia Probability: {prob*100:.2f}%")

        if prob > 0.5:
            st.error("Result: Anemic")
        else:
            st.success("Result: Non-Anemic")
