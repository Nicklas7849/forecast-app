
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta

# Streamlit settings
st.set_page_config(page_title="LSTM Forecast Multi-feature", layout="wide")
st.title("📦 AI Forecast (LSTM) – Multi-feature Efterspørgselsprognose")
st.markdown(
    "Upload dine salgsdata (.csv) med kolonnerne **dato**, **antal_solgt**, **kampagne** og **helligdag**. "
    "Hvis 'kampagne' eller 'helligdag' ikke er til stede, oprettes de med værdi 0."
)

# Upload CSV-fil
uploaded_file = st.file_uploader("Upload CSV-fil", type="csv")

if uploaded_file:
    # Indlæs CSV og konverter dato
    df = pd.read_csv(uploaded_file, parse_dates=['dato'])
    df = df.sort_values('dato')
    
    # Renavn kolonne antal_solgt til 'demand'
    df = df.rename(columns={'antal_solgt': 'demand'})
    
    # Hvis kolonnerne 'kampagne' eller 'helligdag' ikke findes, opret dem som 0
    if 'kampagne' not in df.columns:
        df['kampagne'] = 0
    if 'helligdag' not in df.columns:
        df['helligdag'] = 0

    st.write("**Inputdata:**")
    st.dataframe(df.head(10))
    
    # Vælg features – vi bruger nu 3 variable: demand, kampagne og helligdag
    features = ['demand', 'kampagne', 'helligdag']
    data = df[features].copy()
    
    # Skaler alle features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Lav sekvenser til LSTM
    sequence_length = 10  # Brug de foregående 10 uger til at forudsige næste demand
    X, y = [], []
    # y er target: kun "demand" (den første kolonne i vores matrix)
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Split data (vi træner med ca. 80%)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    # Her kan du også lave en test-split, men i en app er det primært prognose vi vil vise

    # Byg LSTM-model med input shape (sequence_length, antal_features)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    st.info("Modellen trænes – dette kan tage lidt tid...")
    model.fit(X_train, y_train, epochs=100, verbose=0)
    st.success("Modellen er trænet!")

    # Prognose de næste 4 uger
    # Vi starter med den sidste sekvens fra dine data
    last_sequence = scaled_data[-sequence_length:]
    predictions = []
    # Antag, at de eksterne variable (kampagne og helligdag) for fremtiden sættes til 0
    future_external = np.array([0, 0])  # [kampagne, helligdag]
    
    for _ in range(4):
        # Forudsig demand for den næste uge
        pred_scaled = model.predict(last_sequence.reshape(1, sequence_length, X_train.shape[2]), verbose=0)[0][0]
        predictions.append(pred_scaled)
        
        # Lav en ny række med forudsagt demand og standard 0-værdier for de eksterne variable
        new_row = np.concatenate(([pred_scaled], future_external))
        # Opdater last_sequence: fjern den ældste uge og tilføj den nye række
        last_sequence = np.append(last_sequence[1:], [new_row], axis=0)
    
    # Omvend den skalering, men kun for demand (kolonne 0)
    # Vi skal rekonstruere et array, der indeholder samme antal features, men vi kun behøver at invers transformere demand
    pred_array = np.array(predictions).reshape(-1, 1)
    # Find skaleringsparametre for demand: 
    demand_min = scaler.data_min_[0]
    demand_max = scaler.data_max_[0]
    # Invers skalering: x_invers = x_scaled*(max-min) + min
    inversed_pred = pred_array * (demand_max - demand_min) + demand_min

    # Opret fremtidige datoer
    last_date = df['dato'].iloc[-1]
    future_dates = [last_date + timedelta(weeks=i+1) for i in range(4)]
    
    # Lav DataFrame med forecast
    forecast_df = pd.DataFrame({
        'Dato': future_dates,
        'Forventet efterspørgsel': np.round(inversed_pred.flatten()).astype(int)
    })
    
    st.subheader("📊 Prognose – de næste 4 uger")
    st.dataframe(forecast_df)
    
    # Vis graf: Plot historisk demand og forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['dato'], df['demand'], label="Historisk efterspørgsel")
    ax.plot(forecast_df['Dato'], forecast_df['Forventet efterspørgsel'], 
            label="Forecast", linestyle="--", marker='o')
    ax.set_title("Multi-feature LSTM Efterspørgselsprognose")
    ax.set_xlabel("Dato")
    ax.set_ylabel("Efterspørgsel")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Udregn samlet forecast og lav en anbefaling
    total_forecast = int(forecast_df['Forventet efterspørgsel'].sum())
    st.success(f"📦 Anbefaling: Bestil cirka **{total_forecast} stk** de næste 4 uger.")
