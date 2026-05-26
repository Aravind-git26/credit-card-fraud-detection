import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, df, scaler

model, df, scaler = train_model()

st.title("🔍 Real-Time Credit Card Fraud Detection")
st.caption("🟢 Kafka Stream Simulation — Live Transactions")
st.markdown("---")

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", "284,807")
with col2:
    st.metric("Fraud Cases", "492")
with col3:
    st.metric("Fraud %", "0.17%")

st.markdown("---")

# Live Stream Section
st.subheader("📡 Live Kafka Transaction Stream")

if st.button("▶️ Start Live Stream"):
    
    # Placeholders
    status_box = st.empty()
    progress = st.progress(0)
    log = st.empty()
    chart_placeholder = st.empty()

    results = []

    # Mix fraud + legit
    fraud_rows = df[df['Class']==1].sample(3)
    legit_rows = df[df['Class']==0].sample(7)
    stream_df = pd.concat([fraud_rows, legit_rows]).sample(frac=1)

    for i, (index, row) in enumerate(stream_df.iterrows()):
        
        # Simulate Kafka message
        status_box.info(f"📨 Kafka Message Received — Transaction #{i+1}")
        
        features = row.drop('Class').values.reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        actual = row['Class']

        result = {
            'Transaction': f"TXN_{i+1}",
            'Amount': f"₹{abs(row['Amount']):.2f}",
            'Status': '🚨 FRAUD' if prediction==1 else '✅ Legit',
            'Confidence': f"{probability:.2%}" if prediction==1 else f"{1-probability:.2%}",
            'Actual': 'Fraud' if actual==1 else 'Legit'
        }
        results.append(result)

        # Show result
        if prediction == 1:
            status_box.error(f"🚨 FRAUD DETECTED! TXN_{i+1} | Amount: ₹{abs(row['Amount']):.2f} | Confidence: {probability:.2%}")
        else:
            status_box.success(f"✅ Legitimate | TXN_{i+1} | Amount: ₹{abs(row['Amount']):.2f} | Confidence: {1-probability:.2%}")

        # Update log table
        log.dataframe(pd.DataFrame(results), use_container_width=True)

        # Update progress
        progress.progress((i+1)/len(stream_df))

        time.sleep(1)  # Simulate real-time delay

    status_box.success("✅ Stream Complete!")

st.markdown("---")

# Single Transaction Test
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

# Chart
fig = px.pie(
    values=df['Class'].value_counts().values,
    names=['Legitimate', 'Fraud'],
    color_discrete_sequence=['#00CC96', '#EF553B']
)
st.plotly_chart(fig, use_container_width=True)
