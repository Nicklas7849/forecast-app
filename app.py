
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta

# Streamlit settings
st.set_page_config(page_title="LSTM Forecast", layout="wide")
st.title("📦 AI Forecast (LSTM) – Efterspørgselsprognose")
st.markdown("Upload dine salgsdata (.csv) med kolonnerne **dato** og **antal_solgt**")

# Upload fil
uploaded_file = st.file_uploader("Upload CSV-fil", type="csv")

if uploaded_file:
    # Indlæs og klargør data
    df = pd.read_csv(uploaded_file, parse_dates=['dato'])
    df = df.rename(columns={'antal_solgt': 'demand'})
    df = df.sort_values('dato')

    # Skaler efterspørgsel
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['demand']])

    # LSTM-sekvenser
    sequence_length = 10
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    # Træningsdata
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]

    # Byg og træn LSTM-model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)

    # Forudsig næste 4 uger
    last_sequence = scaled_data[-sequence_length:]
    predictions = []
    for _ in range(4):
        pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)[0][0]
        predictions.append(pred)
        last_sequence = np.append(last_sequence[1:], [[pred]], axis=0)

    # Invers skala og lav datoer
    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    last_date = df['dato'].iloc[-1]
    future_dates = [last_date + timedelta(weeks=i+1) for i in range(4)]

    # Tabel
    forecast_df = pd.DataFrame({'Dato': future_dates, 'Forventet efterspørgsel': np.round(forecast).astype(int)})
    st.subheader("📊 Prognose – næste 4 uger")
    st.dataframe(forecast_df)

    # Graf
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['dato'], df['demand'], label="Historisk efterspørgsel")
    ax.plot(future_dates, forecast, label="Forecast", linestyle="--")
    ax.set_title("LSTM-baseret Efterspørgselsprognose")
    ax.set_xlabel("Dato")
    ax.set_ylabel("Efterspørgsel")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Anbefaling
    total = int(sum(forecast))
    st.success(f"📦 Anbefaling: Bestil cirka **{total} stk** de næste 4 uger.")
