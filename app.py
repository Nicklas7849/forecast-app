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

# Sørg for, at st.set_page_config er det allerførste Streamlit‑kald!
st.set_page_config(page_title="Avanceret Forecast", layout="wide")

st.write("Python version:", sys.version)
st.write("TensorFlow version:", tf.__version__)

st.title("📦 AI Forecast (Avanceret) – Efterspørgsels- og Omsætningsprognose")

st.markdown("""
Upload din .csv‑fil med mindst:
- **dato**, **antal_solgt**, **kampagne**, **helligdag**
- Valgfrit: **pris**, **vejr**, **produkt**, **lagerstatus**, **annonceringsomkostning**, **forbrugertillid**
  
Makroøkonomiske variable – **inflation, arbejdsløshed, BNP, rente** – hentes automatisk fra FRED og synkroniseres med dine data.
""")

# --------------------------------------------------------------------------------
# Funktion til at hente nationale økonomiske data fra FRED med forward-fill af kvartalsvise data (BNP)
@st.cache_data(ttl=86400)  # Cache i 24 timer
def get_national_economic_data(api_key, start_date, end_date):
    fred = Fred(api_key=api_key)
    
    def fetch_series(series_id, col_name):
        """
        Hent en enkelt FRED‑serie og returnér den som en DataFrame
        med uger som frekvens ved at forward‑fill de seneste observationer.
        """
        try:
            s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            df_tmp = pd.DataFrame(s, columns=[col_name])
            df_tmp.index = pd.to_datetime(df_tmp.index)
            # Resample til ugentligt med forward‑fill (særligt vigtigt for kvartalsvise data som BNP)
            df_tmp = df_tmp.resample('W').ffill()
            return df_tmp
        except Exception as e:
            st.error(f"Fejl ved hentning af {col_name}: {e}")
            return pd.DataFrame(columns=[col_name], dtype=float)
    
    # FRED‑serier for Danmark:
    # - Inflation (CPI):                    'CPALTT01DKM657N'
    # - Arbejdsløshed:                      'LRHUTTTTDKM156S'
    # - BNP (Real GDP):                       'CLVMNACSCAB1GQDK'    <-- Opdateret BNP‑serie
    # - Rente (10-årig statsobligationsrente): 'IRLTLT01DKM156N'
    df_inflation = fetch_series('CPALTT01DKM657N', 'inflation_fred')
    df_unemployment = fetch_series('LRHUTTTTDKM156S', 'arbejdsløshed_fred')
    df_gdp = fetch_series('CLVMNACSCAB1GQDK', 'BNP_fred')
    df_interest = fetch_series('IRLTLT01DKM156N', 'rente_fred')
    
    # Merge serierne med outer join baseret på dato (ugentligt frekvens)
    df_fred = (df_inflation
               .merge(df_unemployment, how='outer', left_index=True, right_index=True)
               .merge(df_gdp, how='outer', left_index=True, right_index=True)
               .merge(df_interest, how='outer', left_index=True, right_index=True))
    
    df_fred = df_fred.reset_index().rename(columns={'index': 'dato'})
    return df_fred

# --------------------------------------------------------------------------------
# FRED API‑nøgle – enten indtastes af brugeren eller hardcodes til test
fred_api_key = st.text_input("Indtast FRED API‑nøgle (eller indsæt her direkte):", 
                             value="eec7ebe4f9b5cf6161ed55af10fa00c0", type="password")

uploaded_file = st.file_uploader("Upload CSV‑fil", type="csv")

if uploaded_file:
    # Læs CSV‑filen og sorter efter dato
    df = pd.read_csv(uploaded_file, parse_dates=['dato'])
    df = df.sort_values('dato')
    
    # Omdøb 'antal_solgt' til 'demand'
    df = df.rename(columns={'antal_solgt': 'demand'})
    
    # Fjern fallback for inflation, arbejdsløshed, BNP og rente – disse skal synkroniseres.
    # Vi beholder evt. 'pris' og 'forbrugertillid', hvis disse ikke hentes eksternt.
    for col in ['kampagne', 'helligdag', 'vejr', 'lagerstatus', 'annonceringsomkostning', 'pris', 'forbrugertillid']:
        if col not in df.columns:
            df[col] = 0
    
    # Udled tidskomponenter
    df['uge'] = df['dato'].dt.isocalendar().week
    df['måned'] = df['dato'].dt.month
    df['ferie'] = df['måned'].apply(lambda x: 1 if x in [7, 12] else 0)
    
    # Hvis 'produkt' findes, tilbydes et valg
    if 'produkt' in df.columns:
        selected_product = st.selectbox("Vælg produkt", df['produkt'].unique())
        df = df[df['produkt'] == selected_product]
    
    if df.isnull().sum().any():
        st.warning("⚠️ Data indeholder manglende værdier. Kontroller venligst.")
    
    # Hvis en FRED API‑nøgle er angivet, hentes de makroøkonomiske data
    # og de interne værdier for inflation, arbejdsløshed, BNP og rente overskrives.
    if fred_api_key:
        csv_start_date = df['dato'].min().strftime("%Y-%m-%d")
        csv_end_date = df['dato'].max().strftime("%Y-%m-%d")
        df_fred = get_national_economic_data(fred_api_key, csv_start_date, csv_end_date)
        df = pd.merge(df, df_fred, on='dato', how='left')
        for col in ['inflation', 'arbejdsløshed', 'BNP', 'rente']:
            # Sæt de synkroniserede værdier direkte
            df[col] = df[f"{col}_fred"]
    
    # Definér de økonomiske variable, der skal indgå i det aggregerede indeks
    economic_vars = ['pris', 'forbrugertillid', 'inflation', 'arbejdsløshed', 'BNP', 'rente']
    # Sørg for, at alle disse kolonner nu eksisterer (de synkroniserede makrodata skulle fylde inflation, arbejdsløshed, BNP, rente)
    scaler_econ = MinMaxScaler()
    df_econ_scaled = pd.DataFrame(scaler_econ.fit_transform(df[economic_vars]), columns=economic_vars)
    df['økonomisk_indeks'] = df_econ_scaled.mean(axis=1)
    
    st.subheader("📄 Inputdata (første 10 rækker)")
    st.dataframe(df.head(10))
    
    # Indtast fremtidige parametre til forecast
    future_kampagne = st.slider("Fremtidig kampagneintensitet (0-1)", 0.0, 1.0, 0.0, step=0.1)
    future_helligdag = st.slider("Fremtidige helligdage (0-1)", 0.0, 1.0, 0.0, step=0.1)
    tilbudsprocent = st.slider("Tilbudsprocent ved kampagner (%)", 0, 50, 10, step=1)
    
    # Definér features til modellen – her indgår økonomisk_indeks som aggregat for de synkroniserede makrodata
    features = ['demand', 'kampagne', 'helligdag', 'vejr', 'lagerstatus',
                'annonceringsomkostning', 'økonomisk_indeks', 'uge', 'måned', 'ferie']
    data = df[features].copy()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Opret sekvenser til LSTM-modellen
    sequence_length = 10
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # 'demand' er første feature
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
        
        # Forecast: Generer forudsigelser for de næste 12 uger
        forecast_horizon = 12
        last_sequence = scaled_data[-sequence_length:]
        predictions = []
        
        # Dummy-værdier for fremtidige eksterne input: 
        # Vi bruger de angivne slider-værdier for kampagne og helligdag
        # For de øvrige variable (vejr, lagerstatus, annonceringsomkostning) anvendes dummyværdier (0),
        # og økonomisk indeks sættes til gennemsnittet af historiske værdier.
        future_external = np.array([[
            future_kampagne,      # kampagne
            future_helligdag,     # helligdag
            0,                    # vejr (dummy)
            0,                    # lagerstatus (dummy)
            0,                    # annonceringsomkostning (dummy)
            df['økonomisk_indeks'].mean(),  # økonomisk indeks
            0,                    # uge (dummy)
            0,                    # måned (dummy)
            0                     # ferie (dummy)
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

        I denne version hentes makroøkonomiske data for inflation, arbejdsløshed, BNP og rente direkte fra FRED, 
        og eventuelle fallback‑værdier fra CSV‑filen bruges ikke. Derfor afspejler prognosen de seneste, synkroniserede
        nationale nøgletal for Danmark. Ved at integrere disse opdaterede data med historiske salgsdata og en LSTM‑baseret 
        tidsserieanalyse kombineret med en Random Forest-baseline, opnås en mere robust og præcis forecast.
        Kampagneeffekter og dynamisk prisjustering indregnes, hvilket gør prognosen velegnet til strategiske forretningsbeslutninger.
        """)
