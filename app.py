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

# Opsæt Streamlit-siden
st.set_page_config(page_title="Avanceret Forecast", layout="wide")
st.title("📦 AI Forecast (Avanceret) – Efterspørgselsprognose med Økonomiske Indikatorer")

st.markdown("""
Upload din .csv-fil med mindst:
- **dato**, **antal_solgt**, **kampagne**, **helligdag**
- Valgfrit: **pris**, **vejr**, **produkt**, **lagerstatus**, **annonceringsomkostning**, **forbrugertillid**, **inflation**, **arbejdsløshed**, **BNP**, **rente**
""")

uploaded_file = st.file_uploader("Upload CSV-fil", type="csv")

if uploaded_file:
    # Indlæs og sorter data efter dato
    df = pd.read_csv(uploaded_file, parse_dates=['dato'])
    df = df.sort_values('dato')

    # Omdøb kolonnen 'antal_solgt' til 'demand'
    df = df.rename(columns={'antal_solgt': 'demand'})

    # Definer de nødvendige variable – her inddrager vi ud over de oprindelige økonomiske indikatorer også ekstra variable
    required_cols = ['kampagne', 'helligdag', 'pris', 'vejr', 'lagerstatus', 
                     'annonceringsomkostning', 'forbrugertillid', 'inflation',
                     'arbejdsløshed', 'BNP', 'rente']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0  # Hvis kolonnen ikke er til stede, initialiseres den med 0

    # Udled yderligere tidskomponenter
    df['uge'] = df['dato'].dt.isocalendar().week
    df['måned'] = df['dato'].dt.month
    df['ferie'] = df['måned'].apply(lambda x: 1 if x in [7, 12] else 0)

    # Hvis der findes en 'produkt'-kolonne, så tilbydes et valg
    if 'produkt' in df.columns:
        valgte_produkt = st.selectbox("Vælg produkt", df['produkt'].unique())
        df = df[df['produkt'] == valgte_produkt]

    if df.isnull().sum().any():
        st.warning("⚠️ Data indeholder manglende værdier. Kontroller venligst.")

    st.subheader("📄 Inputdata (første 10 rækker)")
    st.dataframe(df.head(10))

    # Konfigurer fremtidige værdier for kampagne og helligdag
    future_kampagne = st.slider("Fremtidig kampagneintensitet (0-1)", 0.0, 1.0, 0.0, step=0.1)
    future_helligdag = st.slider("Fremtidige helligdage (0-1)", 0.0, 1.0, 0.0, step=0.1)

    # Udvid liste af features med de nye økonomiske indikatorer
    features = [
        'demand', 'kampagne', 'helligdag', 'pris', 'vejr', 'lagerstatus', 
        'annonceringsomkostning', 'forbrugertillid', 'inflation',
        'arbejdsløshed', 'BNP', 'rente', 'uge', 'måned', 'ferie'
    ]
    data = df[features].copy()

    # Skaler dataene mellem 0 og 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Opret sekvenser til LSTM-modellen
    sequence_length = 10
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Første element svarer til 'demand'
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        st.error("Ikke nok data til at lave forecast. Tilføj flere rækker til din CSV.")
    else:
        # Split data i træningssæt og test-sæt
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]

        # Definér LSTM-modellen
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

        # Opsætning af forecast-horisont og forberedelse af den sidste kendte sekvens
        forecast_horizon = 12
        last_sequence = scaled_data[-sequence_length:]
        predictions = []
        
        # Byg en fremtidig ekstern input række til variable, 
        # rækkefølgen svarer til features-listen (ekskl. 'demand', som skal forudsiges)
        future_external = np.array([[
            future_kampagne,                # kampagne
            future_helligdag,               # helligdag
            0,                              # pris (kan opdateres hvis forudsigelser for pris er tilgængelige)
            0,                              # vejr
            0,                              # lagerstatus
            0,                              # annonceringsomkostning
            df['forbrugertillid'].mean(),   # forbrugertillid
            df['inflation'].mean(),         # inflation
            df['arbejdsløshed'].mean(),       # arbejdsløshed
            df['BNP'].mean(),               # BNP
            df['rente'].mean(),             # rente
            0,                              # uge (dummy - tidskomponenter bestemmes af dato)
            0,                              # måned (dummy)
            0                               # ferie (dummy)
        ]])

        # Forecast-loop, hvor den forudsagte værdi samles med de eksterne faktorer
        for _ in range(forecast_horizon):
            pred_scaled = model.predict(last_sequence.reshape(1, sequence_length, X.shape[2]), verbose=0)[0][0]
            predictions.append(pred_scaled)
            new_row = np.concatenate(([pred_scaled], future_external.flatten()))
            last_sequence = np.append(last_sequence[1:], [new_row], axis=0)

        # Invers skaler den forudsagte efterspørgsel
        demand_min = scaler.data_min_[0]
        demand_max = scaler.data_max_[0]
        inversed_pred = np.array(predictions) * (demand_max - demand_min) + demand_min

        # Opsæt fremtidige datoer baseret på den sidste dato i datasættet
        last_date = df['dato'].iloc[-1]
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(forecast_horizon)]

        forecast_df = pd.DataFrame({
            'Dato': future_dates,
            'Forventet efterspørgsel': np.round(inversed_pred).astype(int)
        })
        forecast_df['Nedre_grænse'] = (forecast_df['Forventet efterspørgsel'] * 0.85).astype(int)
        forecast_df['Øvre_grænse'] = (forecast_df['Forventet efterspørgsel'] * 1.15).astype(int)

        st.subheader("🔮 Prognose")
        st.dataframe(forecast_df)

        # Udregn et potentielt afkast baseret på gennemsnitsprisen
        potentielt_afkast = forecast_df['Forventet efterspørgsel'].sum() * df['pris'].mean()
        st.write(f"💰 Potentielt afkast ved prognosen: {potentielt_afkast:,.2f} kr.")

        # Plot historisk efterspørgsel og forecast med usikkerhedsinterval
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['dato'], df['demand'], label="Historisk efterspørgsel")
        ax.plot(forecast_df['Dato'], forecast_df['Forventet efterspørgsel'], label="Forecast", linestyle="--", marker='o')
        ax.fill_between(forecast_df['Dato'], forecast_df['Nedre_grænse'], forecast_df['Øvre_grænse'], color='gray', alpha=0.3, label="Usikkerhedsinterval")
        ax.set_title("Avanceret Efterspørgselsprognose")
        ax.set_xlabel("Dato")
        ax.set_ylabel("Efterspørgsel")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.download_button("📥 Download forecast som CSV", forecast_df.to_csv(index=False), file_name="forecast.csv")
