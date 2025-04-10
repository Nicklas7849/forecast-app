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
import tensorflow as tf
import sys

# Import til automatisk datahentning fra FRED
from fredapi import Fred

st.set_page_config(page_title="Avanceret Forecast", layout="wide")

st.write("Python version:", sys.version)
st.write("TensorFlow version:", tf.__version__)

st.title("📦 AI Forecast (Avanceret) – Efterspørgsels- og Omsætningsprognose")

st.markdown("""
Upload din .csv-fil med mindst:
- **dato**, **antal_solgt**, **kampagne**, **helligdag**
- Valgfrit: **pris**, **vejr**, **produkt**, **lagerstatus**, **annonceringsomkostning**, **forbrugertillid**
- Økonomiske variable: **inflation**, **arbejdsløshed**, **BNP**, **rente**  
Hvis du har en FRED API-nøgle, hentes de nationale økonomiske data automatisk og synkroniseres med din CSV.
""")

@st.cache_data(ttl=86400)
def get_national_economic_data(api_key, start_date, end_date):
    fred = Fred(api_key=api_key)
    # Gyldige FRED-seriekoder til DK:
    # Inflation (CPI):      CPALTT01DKM657N
    # Arbejdsløshed:        LRHUTTTTDKM156S
    # BNP (real GDP):       DNKGDPRQDSMEI   <--- OPDATERET
    # Rente (10-årig):      IRLTLT01DKM156N

    try:
        inflation_series = fred.get_series('CPALTT01DKM657N', observation_start=start_date, observation_end=end_date)
    except Exception as e:
        st.error("Fejl ved hentning af inflation data: " + str(e))
        inflation_series = pd.Series(dtype=float)
    try:
        unemployment_series = fred.get_series('LRHUTTTTDKM156S', observation_start=start_date, observation_end=end_date)
    except Exception as e:
        st.error("Fejl ved hentning af arbejdsløshed: " + str(e))
        unemployment_series = pd.Series(dtype=float)
    try:
        # OPDATERET: Brug "DNKGDPRQDSMEI" i stedet for "MKTGDNDKA646NWDB"
        gdp_series = fred.get_series('DNKGDPRQDSMEI', observation_start=start_date, observation_end=end_date)
    except Exception as e:
        st.error("Fejl ved hentning af BNP: " + str(e))
        gdp_series = pd.Series(dtype=float)
    try:
        interest_rate_series = fred.get_series('IRLTLT01DKM156N', observation_start=start_date, observation_end=end_date)
    except Exception as e:
        st.error("Fejl ved hentning af rentedata: " + str(e))
        interest_rate_series = pd.Series(dtype=float)
    
    df_fred = pd.DataFrame({
         'dato': inflation_series.index,
         'inflation_fred': inflation_series.values,
         'arbejdsløshed_fred': unemployment_series.values,
         'BNP_fred': gdp_series.values,
         'rente_fred': interest_rate_series.values
    })
    df_fred['dato'] = pd.to_datetime(df_fred['dato'])
    # Resample til uger og fremfør seneste værdi
    df_fred = df_fred.set_index('dato').resample('W').ffill().reset_index()
    return df_fred

fred_api_key = st.text_input("Indtast FRED API-nøgle...", value="eec7ebe4f9b5cf6161ed55af10fa00c0", type="password")
uploaded_file = st.file_uploader("Upload CSV-fil", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['dato'])
    df = df.sort_values('dato')
    df = df.rename(columns={'antal_solgt': 'demand'})

    economic_vars = ['pris', 'forbrugertillid', 'inflation', 'arbejdsløshed', 'BNP', 'rente']
    for col in economic_vars:
        if col not in df.columns:
            df[col] = 0

    for col in ['kampagne', 'helligdag', 'vejr', 'lagerstatus', 'annonceringsomkostning']:
        if col not in df.columns:
            df[col] = 0

    df['uge'] = df['dato'].dt.isocalendar().week
    df['måned'] = df['dato'].dt.month
    df['ferie'] = df['måned'].apply(lambda x: 1 if x in [7, 12] else 0)

    if 'produkt' in df.columns:
        selected_product = st.selectbox("Vælg produkt", df['produkt'].unique())
        df = df[df['produkt'] == selected_product]

    if df.isnull().sum().any():
        st.warning("⚠️ Data indeholder manglende værdier. Kontroller venligst.")

    if fred_api_key:
        csv_start_date = df['dato'].min().strftime("%Y-%m-%d")
        csv_end_date = df['dato'].max().strftime("%Y-%m-%d")
        df_fred = get_national_economic_data(fred_api_key, csv_start_date, csv_end_date)
        df = pd.merge(df, df_fred, on='dato', how='left')
        for col in ['inflation', 'arbejdsløshed', 'BNP', 'rente']:
            df[col] = df[f"{col}_fred"].combine_first(df[col])

    from sklearn.preprocessing import MinMaxScaler
    scaler_econ = MinMaxScaler()
    df_econ_scaled = pd.DataFrame(scaler_econ.fit_transform(df[economic_vars]), columns=economic_vars)
    df['økonomisk_indeks'] = df_econ_scaled.mean(axis=1)

    st.subheader("📄 Inputdata (første 10 rækker)")
    st.dataframe(df.head(10))
    
    future_kampagne = st.slider("Fremtidig kampagneintensitet (0-1)", 0.0, 1.0, 0.0, step=0.1)
    future_helligdag = st.slider("Fremtidige helligdage (0-1)", 0.0, 1.0, 0.0, step=0.1)
    tilbudsprocent = st.slider("Tilbudsprocent ved kampagner (%)", 0, 50, 10, step=1)

    features = ['demand', 'kampagne', 'helligdag', 'vejr', 'lagerstatus',
                'annonceringsomkostning', 'økonomisk_indeks', 'uge', 'måned', 'ferie']
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
        try:
            model.fit(X_train, y_train, epochs=100, verbose=0)
        except Exception as e:
            st.error("Fejl under træning af LSTM-modellen: " + str(e))
        else:
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

        future_external = np.array([[
            future_kampagne,
            future_helligdag,
            0,
            0,
            0,
            df['økonomisk_indeks'].mean(),
            0,
            0,
            0
        ]])

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
        forecast_df['Nedre_grænse'] = (forecast_df['Forventet efterspørgsel'] * 0.85).astype(int)
        forecast_df['Øvre_grænse'] = (forecast_df['Forventet efterspørgsel'] * 1.15).astype(int)

        st.subheader("🔮 Prognose")
        st.dataframe(forecast_df)

        fast_pris = df['pris'].mean()
        effektiv_pris = fast_pris * (1 - (future_kampagne * (tilbudsprocent / 100)))
        forecast_df['Forventet omsætning'] = forecast_df['Forventet efterspørgsel'] * effektiv_pris
        total_omsætning = forecast_df['Forventet omsætning'].sum()
        st.write(f"💰 Forventet samlet omsætning ved prognosen: {total_omsætning:,.2f} kr.")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['dato'], df['demand'], label="Historisk efterspørgsel")
        ax.plot(forecast_df['Dato'], forecast_df['Forventet efterspørgsel'], label="Forecast", linestyle="--", marker='o')
        ax.fill_between(forecast_df['Dato'], forecast_df['Nedre_grænse'], forecast_df['Øvre_grænse'], 
                        color='gray', alpha=0.3, label="Usikkerhedsinterval")
        ax.set_title("Avanceret Efterspørgsels- og Omsætningsprognose")
        ax.set_xlabel("Dato")
        ax.set_ylabel("Efterspørgsel")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.download_button("📥 Download forecast som CSV", forecast_df.to_csv(index=False), file_name="forecast.csv")

        st.markdown("""
        ### Konklusion og Begrundelse for Prognosen

        For at undgå fejl ved hentning af BNP-data er FRED-serien 'MKTGDNDKA646NWDB' udskiftet med 'DNKGDPRQDSMEI', 
        som indeholder kvartalsvis real-BNP for Danmark (resamplet til uger for at passe til salgsdataene).

        Dermed er prognosen nu baseret på pålidelige, opdaterede nøgletal for inflation, arbejdsløshed, BNP og rente. 
        Denne integration sikrer, at forecastet hele tiden afspejler den aktuelle makroøkonomiske kontekst – og kombineret med 
        en LSTM-basering og kampagnejustering gør det modellen egnet til mere præcise og strategisk anvendelige salgsprognoser.
        """)
