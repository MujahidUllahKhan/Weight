# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression

MODEL = "bmi_cat_model.pkl"

# ---------- 1.  TRAIN / LOAD ----------
if not os.path.exists(MODEL):
    df = pd.DataFrame({"bmi": [16, 20, 24, 28, 32],
                       "label": [0, 1, 1, 2, 3]})  # 0 under, 1 normal, 2 over, 3 obese
    clf = LogisticRegression(max_iter=1000).fit(df[["bmi"]], df["label"])
    joblib.dump(clf, MODEL)
else:
    clf = joblib.load(MODEL)

# ---------- 2.  UI ----------
st.set_page_config(page_title="BMI Checker", page_icon="⚖️")
st.title("⚖️ BMI Category Predictor")

c1, c2 = st.columns(2)
with c1:
    w = st.number_input("Weight (kg)", 1.0, 300.0, 70.0, 0.5)
with c2:
    h_cm = st.number_input("Height (cm)", 50, 250, 170, 1)

if st.button("Predict"):
    h_m   = h_cm / 100
    bmi   = w / (h_m ** 2)
    proba = clf.predict_proba([[bmi]])[0]
    cat   = clf.classes_[proba.argmax()]
    labels = {0: "Underweight 🟡", 1: "Normal 🟢", 2: "Overweight 🟠", 3: "Obese 🔴"}
    st.success(f"BMI = **{bmi:.1f}**  |  Category: **{labels[cat]}**")
    st.bar_chart({"Probability": proba})
