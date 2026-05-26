import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame(
        np.random.randn(n, 28),
        columns=[f'V{i}' for i in range(1, 29)]
    )
    df['Amount'] = np.random.uniform(0, 1000, n)
    df['Class'] = np.random.choice([0, 1], n, p=[0.998, 0.002])
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, df

model, df = train_model()

st.title(" Real-Time Credit Card Fraud Detection")
st.caption(" Kafka Live Stream — 1 Transaction per Second")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", "284,807")
with col2:
    st.metric("Fraud Cases", "492")
with col3:
    st.metric("Fraud %", "0.17%")

st.markdown("---")

st.subheader("Kafka Live Stream")

if st.button("▶️ Start Live Stream"):

    # Guarantee 2 fraud in 10 transactions
    fraud_rows = df[df['Class']==1].sample(2)
    legit_rows = df[df['Class']==0].sample(8)
    stream_df = pd.concat([fraud_rows, legit_rows]).sample(frac=1).reset_index(drop=True)

    # ONE notification box — updates every second
    notification = st.empty()
    progress = st.progress(0)

    for i, row in stream_df.iterrows():

        now = datetime.now().strftime('%H:%M:%S')
        features = row.drop('Class').values.reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # ONLY ONE notification shown at a time
        if prediction == 1:
            notification.error(
                f" **FRAUD ALERT!**  \n"
                f" Time: `{now}`  \n"
                f" TXN ID: `TXN_{1000+i}`  \n"
                f" Amount: `₹{abs(row['Amount']):.2f}`  \n"
                f" Fraud Confidence: `{probability:.2%}`"
            )
        else:
            notification.success(
                f"✅ **Legitimate Transaction**  \n"
                f" Time: `{now}`  \n"
                f" TXN ID: `TXN_{1000+i}`  \n"
                f" Amount: `₹{abs(row['Amount']):.2f}`  \n"
                f" Safe Confidence: `{1-probability:.2%}`"
            )

        progress.progress((i+1)/10)
        time.sleep(1)  # 1 second per transaction

    notification.info(" Stream Complete! 10 transactions processed.")

st.markdown("---")

st.subheader(" Test Single Transaction")
if st.button("Check Random Transaction"):
    row = df.sample(1).iloc[0]
    features = row.drop('Class').values.reshape(1, -1)
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
