
import streamlit as st
import pandas as pd, numpy as np, joblib, os
from sklearn.linear_model import LogisticRegression

MODEL = "bmi_cat_model.pkl"
if not os.path.exists(MODEL):
    df = pd.DataFrame({"bmi":[16,20,24,28,32], "label":[0,1,1,2,3]})
    clf = LogisticRegression().fit(df[["bmi"]], df["label"])
    joblib.dump(clf, MODEL)
else:
    clf = joblib.load(MODEL)

st.title("Colab → Streamlit demo")
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
if st.button("Predict"):
    cat = clf.predict([[bmi]])[0]
    st.success(f"Category {cat}")

!nohup streamlit run app.py &>log.txt &
!sleep 5 && curl -s http://localhost:8501 > /dev/null && echo "Streamlit alive"
