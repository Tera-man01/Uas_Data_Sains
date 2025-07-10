import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Data mini latih (atau load dari CSV jika tersedia)
X = np.array([
    [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6],
    [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3],
    [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7],
    [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0, 3],
    [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, 3],
    [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 3]
])
y = np.array([1, 0, 0, 1, 1, 1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

st.title("💓 Prediksi Penyakit Jantung")

# Form input
age = st.number_input("Umur", 20, 100, 50)
sex = st.radio("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
cp = st.selectbox("Tipe Nyeri Dada (1–4)", [1, 2, 3, 4])
trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 120)
chol = st.number_input("Kolesterol", 100, 400, 200)
fbs = st.radio("Gula Darah > 120 mg/dl", [1, 0])
restecg = st.selectbox("Hasil EKG (0–2)", [0, 1, 2])
thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150)
exang = st.radio("Angina karena Olahraga", [1, 0])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Kemiringan ST (1–3)", [1, 2, 3])
ca = st.selectbox("Jumlah Pembuluh (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [3, 6, 7])

if st.button("🔍 Prediksi"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success("✅ Hasil: Kemungkinan memiliki penyakit jantung.")
    else:
        st.info("🫀 Hasil: Kemungkinan tidak memiliki penyakit jantung.")

    st.write(f"📊 Probabilitas: {prob*100:.2f}%")
