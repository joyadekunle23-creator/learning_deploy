import joblib
import numpy as np
import streamlit as st

model = joblib.load(open("titanic_model_plain.pkl", "rb"))

st.title("Titanic Survival Predictor")

pclass = st.selectbox("Passenger class", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.number_input("Age", 0.0, 100.0, 30.0)
sibsp = st.number_input("Siblings/spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of embarkation", ["C", "Q", "S"])

sex_val = 0 if sex == "female" else 1
emb_val = {"C": 0, "Q": 1, "S": 2}[embarked]

if st.button("Predict"):
    X = np.array([[pclass, sex_val, age, sibsp, parch, fare, emb_val]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    if pred == 1:
        st.success(f"Likely survived (probability {prob:.0%})")
    else:
        st.error(f"Likely did not survive (probability {prob:.0%})")