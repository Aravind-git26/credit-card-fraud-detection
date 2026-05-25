import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

@st.cache_resource
def load_model():
    with open('Fraud_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame(
        np.random.randn(n, 28),
        columns=[f'V{i}' for i in range(1, 29)]
    )
    df['Amount'] = np.random.uniform(0, 1000, n)
    df['Time'] = np.random.uniform(0, 172800, n)
    df['Class'] = np.random.choice([0, 1], n, p=[0.998, 0.002])
    return df

model = load_model()
df = load_data()

st.title("🔍 Credit Card Fraud Detection")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", "284,807")
with col2:
    st.metric("Fraud Cases", "492")
with col3:
    st.metric("Fraud %", "0.17%")

st.markdown("---")

st.subheader("🎯 Test Random Transaction")
if st.button("Check Random Transaction"):
    row = df.sample(1).iloc[0]
    features = row.drop(['Class', 'Time']).values.reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    actual = row['Class']
    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED! Confidence: {probability:.2%}")
    else:
        st.success(f"✅ Legitimate! Confidence: {1-probability:.2%}")
    st.info(f"Actual: {'Fraud' if actual==1 else 'Legitimate'}")

st.markdown("---")
fig = px.pie(
    values=df['Class'].value_counts().values,
    names=['Legitimate', 'Fraud'],
    color_discrete_sequence=['#00CC96', '#EF553B']
)
st.plotly_chart(fig, use_container_width=True)
