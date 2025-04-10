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

# Sørg for, at st.set_page_config er det allerførste Streamlit-kald!
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

@st.cache_data(ttl=86400)  # Cache i 24 timer
def get_national_economic_data(api_key, start_date, end_date):
    fred = Fred(api_key=api_key)
    
    def fetch_series(series_id, col_name):
        try:
            s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            df_tmp = pd.DataFrame(s, columns=[col_name])
            df_tmp.index = pd.to_datetime(df_tmp.index)
            df_tmp = df_tmp.resample('W').ffill()  # Forward-fill ved ugentlige data
            return df_tmp
        except Exception as e:
            st.error(f"Fejl ved hentning af {col_name}: {e}")
            return pd.DataFrame(columns=[col_name], dtype=float)
    
    df_inflation = fetch_series('CPALTT01DKM657N', 'inflation_fred')
    df_unemployment = fetch_series('LRHUTTTTDKM156S', 'arbejdsløshed_fred')
    df_gdp = fetch_series('CLVMNACSCAB1GQDK', 'BNP_fred')
    df_interest = fetch_series('IRLTLT01DKM156N', 'rente_fred')

    df_fred = (df_inflation
               .merge(df_unemployment, how='outer', left_index=True, right_index=True)
               .merge(df_gdp, how='outer', left_index=True, right_index=True)
               .merge(df_interest, how='outer', left_index=True, right_index=True))

    df_fred = df_fred.reset_index().rename(columns={'index': 'dato'})
    return df_fred

fred_api_key = st.text_input("Indtast FRED API‑nøgle (eller indsæt her direkte):", 
                             value="eec7ebe4f9b5cf6161ed55af10fa00c0", type="password")

uploaded_file = st.file_uploader("Upload CSV‑fil", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['dato'])
    df = df.sort_values('dato')
    
    df = df.rename(columns={'antal_solgt': 'demand'})
    
    # Tjekker for kolonner – inflation, arbejdsløshed, BNP og rente hentes KUN fra FRED
    for col in ['kampagne', 'helligdag', 'vejr', 'lagerstatus', 'annonceringsomkostning', 'pris', 'forbrugertillid']:
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
            df[col] = df[f"{col}_fred"]

    economic_vars = ['pris', 'forbrugertillid', 'inflation', 'arbejdsløshed', 'BNP', 'rente']
    scaler_econ = MinMaxScaler()
    df_econ_scaled = pd.DataFrame(scaler_econ.fit_transform(df[economic_vars]), columns=economic_vars)
    df['økonomisk_indeks'] = df_econ_scaled.mean(axis=1)

    st.subheader("📄 Inputdata (første 10 rækker)")
    st.dataframe(df.head(10))
    
    future_kampagne = st.slider("Fremtidig kampagneintensitet (0-1)", 0.0, 1.0, 0.0, step=0.1)
    future_helligdag = st.slider("Fremtidige helligdage (0-1)", 0.0, 1.0, 0.0, step=0.1)
    tilbudsprocent = st.slider("Tilbudsprocent ved kampagner (%)", 0, 50, 10, step=1)
    shock_percent = st.slider("Uforudsigelige Stød (%):", -20, 20, 0, step=1)

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

        inversed_pred_adjusted = inversed_pred * (1 + shock_percent / 100.0)

        last_date = df['dato'].iloc[-1]
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(forecast_horizon)]
        forecast_df = pd.DataFrame({
            'Dato': future_dates,
            'Forventet efterspørgsel': np.round(inversed_pred_adjusted).astype(int)
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
        ax.plot(forecast_df['Dato'], forecast_df['Forventet efterspørgsel'], 
                label="Forecast", linestyle="--", marker='o')
        ax.fill_between(forecast_df['Dato'], forecast_df['Nedre_grænse'], forecast_df['Øvre_grænse'], 
                        color='gray', alpha=0.3, label="Usikkerhedsinterval")
        ax.set_title("Avanceret Efterspørgsels- og Omsætningsprognose")
        ax.set_xlabel("Dato")
        ax.set_ylabel("Efterspørgsel")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.download_button("📥 Download forecast som CSV", forecast_df.to_csv(index=False), file_name="forecast.csv")

        # ---------------------------------------------------------------
        # Tilføjet Konklusion og korrigeret matematik i laTex-format
        st.markdown(r"""
### Konklusion
Modellen integrerer historiske salgsdata med automatiske makrodata fra FRED og en 
uforudsigelig stød-konstant. Ved at anvende en LSTM-baseret tidsserieanalyse 
kombineret med en Random Forest-baseline samt en dynamisk prisjustering, 
får vi en solid prognose, der kan tilpasses pludselige begivenheder i markedet.

### Matematik og Fortolkning

**1. Dataforbehandling og Normalisering**  
Før modellering skaleres alle inputvariable til intervallet \([0, 1]\) ved hjælp af Min-Max scaling:

$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

**2. LSTM-Modellen**  
LSTM-cellen opererer med følgende matematiske ligninger:

$$
\begin{aligned}
i_t &= \sigma(W_{xi}\, x_t + W_{hi}\, h_{t-1} + b_i),\\
f_t &= \sigma(W_{xf}\, x_t + W_{hf}\, h_{t-1} + b_f),\\
g_t &= \tanh(W_{xg}\, x_t + W_{hg}\, h_{t-1} + b_g),\\
o_t &= \sigma(W_{xo}\, x_t + W_{ho}\, h_{t-1} + b_o),\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t,\\
h_t &= o_t \odot \tanh(c_t).
\end{aligned}
$$

Her er:
- \(x_t\) input på tid \(t\),
- \(h_t\) skjult tilstand,
- \(c_t\) celle-tilstand,
- \(\sigma\) sigmoidfunktionen,
- \(\tanh\) hyperbolsk tangent,
- \(\odot\) elementvis multiplikation.

Den endelige forudsigelse fås ved at føre den sidste skjulte tilstand \(h_t\) gennem et Dense-lag:

$$
\hat{y} = W_{\text{dense}}\, h_t + b_{\text{dense}}
$$

**3. Random Forest Baseline**  
En Random Forest Regressor trænes som baseline. Den anvender \(\text{n}\) beslutningstræer 
til at minimere fejl og sammenlignes med LSTM-modellen for at vurdere gevinsten ved LSTM.

**4. Forecast Beregning og Inverse Skalering**  
Efter LSTM-modellens forudsigelse af den normaliserede efterspørgsel \(\hat{y}_{\text{norm}}\), 
transformeres den til den oprindelige skala:

$$
\hat{y} = \hat{y}_{\text{norm}} \times \bigl[\max(\text{demand}) - \min(\text{demand})\bigr] + \min(\text{demand})
$$

**5. Uforudsigelige Stød**  
For at tage højde for uforudsigelige begivenheder anvendes en shock-faktor \(S\) i procent:

$$
\hat{y}_{\text{adjusted}} = \hat{y} \times \Bigl(1 + \frac{S}{100}\Bigr)
$$

Dette gør det muligt at simulere fx en økonomisk krise eller en uventet kampagneeffekt.

**6. Beregning af Effektiv Pris og Omsætning**  
Prisen justeres med en rabat, som afhænger af kampagneintensiteten \(\alpha\) og tilbudsprocent \(\beta\):

$$
\text{Effektiv pris} = \text{Fast pris} \times \Bigl(1 - \alpha \times \frac{\beta}{100}\Bigr)
$$

Omsætningen fås som:

$$
\text{Omsætning} = \hat{y}_{\text{adjusted}} \times \text{Effektiv pris}
$$

**Fortolkning**  
- *Inputdata* integrerer både interne salgsfaktorer og eksterne makroforhold.  
- *LSTM* fanger tidslig dynamik og sæsonmønstre i efterspørgslen.  
- *Uforudsigelige Stød* muliggør manuel justering for uventede hændelser.  
- *Kampagne- og prisdata* omsættes direkte til en økonomisk vurdering af effekten.  

Samlet set giver modellen en robust prognose, der er fleksibel nok 
til at håndtere både planlagte ændringer (kampagner, prisjusteringer) 
og pludselige begivenheder.
""")
