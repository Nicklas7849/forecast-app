# Udvidet version af din forecast-app med forbedringer

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Avanceret Forecast", layout="wide")
st.title("📦 AI Forecast (Avanceret) – Efterspørgselsprognose med ekstra variable")

st.markdown("""
Upload din .csv-fil med mindst:
- **dato**, **antal_solgt**, **kampagne**, **helligdag**
- Valgfrit: **pris**, **vejr**, **produkt**
""")

uploaded_file = st.file_uploader("Upload CSV-fil", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['dato'])
    df = df.sort_values('dato')

    df = df.rename(columns={'antal_solgt': 'demand'})

    for col in ['kampagne', 'helligdag', 'pris', 'vejr']:
        if col not in df.columns:
            df[col] = 0

    df['uge'] = df['dato'].dt.isocalendar().week
    df['måned'] = df['dato'].dt.month
    df['år'] = df['dato'].dt.year

    if 'produkt' in df.columns:
        valgte_produkt = st.selectbox("Vælg produkt", df['produkt'].unique())
        df = df[df['produkt'] == valgte_produkt]

    st.subheader("📄 Inputdata (første 10 rækker)")
    st.dataframe(df.head(10))

    future_kampagne = st.slider("Fremtidig kampagneintensitet (0-1)", 0.0, 1.0, 0.0, step=0.1)
    future_helligdag = st.slider("Fremtidige helligdage (0-1)", 0.0, 1.0, 0.0, step=0.1)

    features = ['demand', 'kampagne', 'helligdag', 'pris', 'vejr', 'uge', 'måned']
    data = df[features].copy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence_length = 10
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        st.error("Ikke nok data til at lave forecast. Tilføj flere rækker til din CSV.")
    else:
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        st.info("Træner LSTM-model...")
        model.fit(X_train, y_train, epochs=100, verbose=0)
        st.success("✅ LSTM færdigtrænet")

        st.info("Træner Random Forest (baseline)...")
        rf = RandomForestRegressor()
        rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        rf_pred = rf.predict(X_train.reshape(X_train.shape[0], -1))
        rf_mse = mean_squared_error(y_train, rf_pred)
        st.write(f"🌲 Random Forest MSE (train): {rf_mse:.2f}")

        forecast_horizon = 12
        last_sequence = scaled_data[-sequence_length:]
        predictions = []
        future_external = np.array([[future_kampagne, future_helligdag, 0, 0, 0, 0]])

        for _ in range(forecast_horizon):
            pred_scaled = model.predict(last_sequence.reshape(1, sequence_length, X.shape[2]), verbose=0)[0][0]
            predictions.append(pred_scaled)
            new_row = np.concatenate(([pred_scaled], future_external.flatten()))
            last_sequence = np.append(last_sequence[1:], [new_row], axis=0)

        demand_min = scaler.data_min_[0]
        demand_max = scaler.data_max_[0]
        inversed_pred = np.array(predictions) * (demand_max - demand_min) + demand_min

        last_date = df['dato'].iloc[-1]
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(forecast_horizon)]

        forecast_df = pd.DataFrame({
            'Dato': future_dates,
            'Forventet efterspørgsel': np.round(inversed_pred).astype(int)
        })

        st.subheader("🔮 Prognose")
        st.dataframe(forecast_df)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['dato'], df['demand'], label="Historisk efterspørgsel")
        ax.plot(forecast_df['Dato'], forecast_df['Forventet efterspørgsel'], label="Forecast", linestyle="--", marker='o')
        ax.set_title("Avanceret Efterspørgselsprognose")
        ax.set_xlabel("Dato")
        ax.set_ylabel("Efterspørgsel")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        total_forecast = int(forecast_df['Forventet efterspørgsel'].sum())
        seneste_efterspørgsel = df['demand'].iloc[-1]
        forventet_uge_1 = forecast_df['Forventet efterspørgsel'].iloc[0]
        ændring = forventet_uge_1 - seneste_efterspørgsel
        seneste_kampagner = df['kampagne'].tail(10).sum()
        seneste_helligdage = df['helligdag'].tail(10).sum()

        forklaring = f"""
📈 **Anbefaling: Bestil cirka {total_forecast} stk de næste 12 uger.**

- Seneste kendte efterspørgsel: {seneste_efterspørgsel} stk
- Forventet uge 1: {forventet_uge_1} stk
- {'Stigning' if ændring > 0 else 'Reduktion'} på {abs(ændring)} stk

Modellen tager højde for kampagner, sæsonmønstre, pris og vejrpåvirkning i data.
"""
        if seneste_kampagner:
            forklaring += f"- {seneste_kampagner} kampagner påvirkede de sidste uger\n"
        if seneste_helligdage:
            forklaring += f"- {seneste_helligdage} helligdage registreret i perioden\n"

        forklaring += """

✅ Prognosen er baseret på 10 ugers historik, mønstre og variabler, og kan justeres med nye input.
        """
        st.markdown(forklaring)
        st.download_button("📥 Download forecast som CSV", forecast_df.to_csv(index=False), file_name="forecast.csv")
