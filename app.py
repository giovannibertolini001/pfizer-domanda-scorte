import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    ExponentialSmoothing
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# =========================================================
# CONFIGURAZIONE
# =========================================================

st.set_page_config(
    page_title="Pfizer - Domanda e Scorte",
    page_icon="💊",
    layout="wide"
)


# =========================================================
# PRODOTTI
# =========================================================

prodotti = {
    "Zithromax": {
        "media": 8000,
        "variabilita": 0.12,
        "stagionalita": 0.25,
        "trend": 0,
        "lead_time": 2,
        "servizio": 0.95,
        "quantita_ordine": 16000
    },

    "Vfend": {
        "media": 2500,
        "variabilita": 0.07,
        "stagionalita": 0.03,
        "trend": 0.01,
        "lead_time": 3,
        "servizio": 0.95,
        "quantita_ordine": 7500
    },

    "Paxlovid": {
        "media": 3500,
        "variabilita": 0.30,
        "stagionalita": 0.10,
        "trend": 0,
        "lead_time": 1,
        "servizio": 0.95,
        "quantita_ordine": 7000
    },

    "Prevnar 20": {
        "media": 4500,
        "variabilita": 0.10,
        "stagionalita": 0.30,
        "trend": 0.02,
        "lead_time": 4,
        "servizio": 0.98,
        "quantita_ordine": 18000
    },

    "Abrysvo": {
        "media": 3000,
        "variabilita": 0.12,
        "stagionalita": 0.45,
        "trend": 0.03,
        "lead_time": 3,
        "servizio": 0.98,
        "quantita_ordine": 12000
    },

    "Eliquis": {
        "media": 12000,
        "variabilita": 0.05,
        "stagionalita": 0.02,
        "trend": 0.08,
        "lead_time": 2,
        "servizio": 0.95,
        "quantita_ordine": 24000
    }
}


# =========================================================
# FUNZIONE GENERAZIONE DOMANDA
# =========================================================

def genera_domanda(
    media,
    variabilita,
    stagionalita,
    trend,
    settimane=156,
    picchi=False,
    seed=None
):
    rng = np.random.default_rng(seed)

    t = np.arange(settimane)

    componente_trend = (
        media
        * trend
        * (t / settimane)
    )

    componente_stagionale = (
        media
        * stagionalita
        * np.sin(2 * np.pi * t / 52)
    )

    componente_casuale = rng.normal(
        0,
        media * variabilita,
        settimane
    )

    domanda = (
        media
        + componente_trend
        + componente_stagionale
        + componente_casuale
    )

    if picchi:
        numero_picchi = 4

        settimane_picco = rng.choice(
            settimane,
            size=numero_picchi,
            replace=False
        )

        for settimana in settimane_picco:
            domanda[settimana] *= rng.uniform(1.5, 2.2)

    return np.maximum(domanda, 0)


# =========================================================
# PREVISIONE MEDIA MOBILE
# =========================================================

def previsione_media_mobile(
    train,
    numero_periodi=12,
    finestra=4
):
    storico = list(train)
    previsioni = []

    for _ in range(numero_periodi):
        previsione = np.mean(storico[-finestra:])
        previsioni.append(previsione)
        storico.append(previsione)

    return np.array(previsioni)


# =========================================================
# EXPONENTIAL SMOOTHING
# =========================================================

def previsione_exponential_smoothing(
    train,
    numero_periodi=12
):
    modello = SimpleExpSmoothing(
        train,
        initialization_method="estimated"
    ).fit(
        optimized=True
    )

    previsione = modello.forecast(
        numero_periodi
    )

    return np.array(previsione)


# =========================================================
# HOLT-WINTERS
# =========================================================

def previsione_holt_winters(
    train,
    numero_periodi=12
):
    modello = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=52
    ).fit(
        optimized=True
    )

    previsione = modello.forecast(
        numero_periodi
    )

    return np.maximum(
        np.array(previsione),
        0
    )


# =========================================================
# METRICHE
# =========================================================

def calcola_metriche(
    reale,
    previsto
):
    mae = mean_absolute_error(
        reale,
        previsto
    )

    rmse = np.sqrt(
        mean_squared_error(
            reale,
            previsto
        )
    )

    reale_array = np.array(reale)
    previsto_array = np.array(previsto)

    valori_validi = reale_array != 0

    mape = np.mean(
        np.abs(
            (
                reale_array[valori_validi]
                - previsto_array[valori_validi]
            )
            / reale_array[valori_validi]
        )
    ) * 100

    return mae, rmse, mape


# =========================================================
# CALCOLO SCORTE
# =========================================================

def calcola_scorte(
    domanda_media,
    deviazione_standard,
    lead_time,
    livello_servizio,
    quantita_ordine
):
    valori_z = {
        0.90: 1.28,
        0.95: 1.65,
        0.98: 2.05,
        0.99: 2.33
    }

    z = valori_z[livello_servizio]

    safety_stock = (
        z
        * deviazione_standard
        * np.sqrt(lead_time)
    )

    cycle_stock = (
        quantita_ordine / 2
    )

    anticipated_stock = (
        domanda_media
        * lead_time
        * 0.20
    )

    pipeline_stock = (
        domanda_media
        * lead_time
    )

    average_stock = (
        cycle_stock
        + safety_stock
        + anticipated_stock
        + pipeline_stock
    )

    reorder_point = (
        domanda_media * lead_time
        + safety_stock
    )

    return {
        "Safety Stock": safety_stock,
        "Cycle Stock": cycle_stock,
        "Anticipated Stock": anticipated_stock,
        "Pipeline Stock": pipeline_stock,
        "Average Stock": average_stock,
        "Reorder Point": reorder_point
    }


# =========================================================
# TITOLO DASHBOARD
# =========================================================

st.title(
    "Pfizer - Analisi della domanda e delle scorte"
)

st.write(
    "Dashboard per la simulazione della domanda, "
    "la previsione e la gestione delle scorte."
)

st.caption(
    "I dati utilizzati sono simulati a fini didattici "
    "e non rappresentano dati reali di vendita."
)


# =========================================================
# SELEZIONE PRODOTTO
# =========================================================

prodotto = st.selectbox(
    "Seleziona il prodotto",
    list(prodotti.keys())
)

parametri = prodotti[prodotto]


# =========================================================
# PARAMETRI
# =========================================================

st.subheader(
    "Parametri della domanda e dell'approvvigionamento"
)

col1, col2, col3 = st.columns(3)

with col1:
    media = st.number_input(
        "Domanda media",
        min_value=100,
        value=parametri["media"],
        step=100,
        key=f"media_{prodotto}"
    )

    variabilita = st.slider(
        "Variabilità",
        0.0,
        0.50,
        float(parametri["variabilita"]),
        0.01,
        key=f"variabilita_{prodotto}"
    )

    stagionalita = st.slider(
        "Stagionalità",
        0.0,
        0.60,
        float(parametri["stagionalita"]),
        0.01,
        key=f"stagionalita_{prodotto}"
    )

with col2:
    trend = st.slider(
        "Trend",
        -0.10,
        0.10,
        float(parametri["trend"]),
        0.01,
        key=f"trend_{prodotto}"
    )

    lead_time = st.number_input(
        "Lead Time (settimane)",
        min_value=1,
        max_value=12,
        value=parametri["lead_time"],
        key=f"lead_time_{prodotto}"
    )

with col3:
    servizio = st.selectbox(
        "Livello di servizio",
        [0.90, 0.95, 0.98, 0.99],
        index=[
            0.90,
            0.95,
            0.98,
            0.99
        ].index(
            parametri["servizio"]
        ),
        format_func=lambda x: f"{x * 100:.0f}%",
        key=f"servizio_{prodotto}"
    )

    quantita_ordine = st.number_input(
        "Quantità d'ordine",
        min_value=100,
        value=parametri["quantita_ordine"],
        step=500,
        key=f"ordine_{prodotto}"
    )


# =========================================================
# GENERAZIONE DOMANDA
# =========================================================

if "seed" not in st.session_state:
    st.session_state.seed = 42

if st.button(
    "🔄 Genera nuova domanda"
):
    st.session_state.seed += 1

seed = st.session_state.seed

domanda = genera_domanda(
    media=media,
    variabilita=variabilita,
    stagionalita=stagionalita,
    trend=trend,
    settimane=156,
    picchi=(prodotto == "Paxlovid"),
    seed=seed
)


# =========================================================
# SERIE TEMPORALE
# =========================================================

serie = pd.Series(domanda)

train = serie.iloc[:144]
test = serie.iloc[144:]

date = pd.date_range(
    start="2023-01-02",
    periods=156,
    freq="W-MON"
)

date_test = date[-12:]


# =========================================================
# PREVISIONI
# =========================================================

previsione_mm = previsione_media_mobile(train)
previsione_es = previsione_exponential_smoothing(train)
previsione_hw = previsione_holt_winters(train)


# =========================================================
# METRICHE
# =========================================================

metriche_mm = calcola_metriche(
    test,
    previsione_mm
)

metriche_es = calcola_metriche(
    test,
    previsione_es
)

metriche_hw = calcola_metriche(
    test,
    previsione_hw
)
metodi = {
    "Media Mobile": metriche_mm[2],
    "Exponential Smoothing": metriche_es[2],
    "Holt-Winters": metriche_hw[2]
}

metodo_migliore = min(metodi, key=metodi.get)

st.success(
    f"Metodo più adatto per {prodotto}: {metodo_migliore} "
    f"(MAPE = {metodi[metodo_migliore]:.2f}%)"
)

# =========================================================
# GRAFICI DELLE PREVISIONI
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("1. Media Mobile")

    fig1, ax1 = plt.subplots(
        figsize=(6, 2.5)
    )

    ax1.plot(
        date_test,
        test.values,
        label="Domanda reale"
    )

    ax1.plot(
        date_test,
        previsione_mm,
        label="Media Mobile"
    )

    ax1.set_xlabel("Data")
    ax1.set_ylabel("Confezioni")
    ax1.legend()
    ax1.grid(True)

    st.pyplot(fig1)


with col2:

    st.subheader("2. Exponential Smoothing")

    fig2, ax2 = plt.subplots(
        figsize=(6, 2.5)
    )

    ax2.plot(
        date_test,
        test.values,
        label="Domanda reale"
    )

    ax2.plot(
        date_test,
        previsione_es,
        label="Exponential Smoothing"
    )

    ax2.set_xlabel("Data")
    ax2.set_ylabel("Confezioni")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)


col3, col4 = st.columns(2)

with col3:

    st.subheader("3. Holt-Winters")

    fig3, ax3 = plt.subplots(
        figsize=(6, 2.5)
    )

    ax3.plot(
        date_test,
        test.values,
        label="Domanda reale"
    )

    ax3.plot(
        date_test,
        previsione_hw,
        label="Holt-Winters"
    )

    ax3.set_xlabel("Data")
    ax3.set_ylabel("Confezioni")
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)


# =========================================================
# ACCURATEZZA
# =========================================================

st.subheader("Accuratezza delle previsioni")

col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig1, use_container_width=True)

with col2:
    st.pyplot(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.pyplot(fig3, use_container_width=True)

# =========================================================
# SCORTE
# =========================================================

deviazione_base = (
    parametri["media"]
    * parametri["variabilita"]
)

scorte_base = calcola_scorte(
    domanda_media=parametri["media"],
    deviazione_standard=deviazione_base,
    lead_time=parametri["lead_time"],
    livello_servizio=parametri["servizio"],
    quantita_ordine=parametri["quantita_ordine"]
)

deviazione_attuale = (
    media
    * variabilita
)

scorte_attuali = calcola_scorte(
    domanda_media=media,
    deviazione_standard=deviazione_attuale,
    lead_time=lead_time,
    livello_servizio=servizio,
    quantita_ordine=quantita_ordine
)


# =========================================================
# VISUALIZZAZIONE SCORTE
# =========================================================

st.subheader("Gestione delle scorte")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Safety Stock",
        f"{scorte_attuali['Safety Stock']:.0f}"
    )

with col2:
    st.metric(
        "Cycle Stock",
        f"{scorte_attuali['Cycle Stock']:.0f}"
    )

with col3:
    st.metric(
        "Anticipated Stock",
        f"{scorte_attuali['Anticipated Stock']:.0f}"
    )

with col4:
    st.metric(
        "Pipeline Stock",
        f"{scorte_attuali['Pipeline Stock']:.0f}"
    )

st.write(
    f"Average Stock: {scorte_attuali['Average Stock']:.0f} | "
    f"Reorder Point: {scorte_attuali['Reorder Point']:.0f}"
)

# =========================================================
# CONFRONTO PRIMA / DOPO
# =========================================================

st.subheader("Effetto della modifica dei parametri")

confronto = pd.DataFrame({
    "Indicatore": [
        "Safety Stock",
        "Cycle Stock",
        "Average Stock",
        "Reorder Point"
    ],

    "Valore iniziale": [
        scorte_base["Safety Stock"],
        scorte_base["Cycle Stock"],
        scorte_base["Average Stock"],
        scorte_base["Reorder Point"]
    ],

    "Valore attuale": [
        scorte_attuali["Safety Stock"],
        scorte_attuali["Cycle Stock"],
        scorte_attuali["Average Stock"],
        scorte_attuali["Reorder Point"]
    ]
})

confronto["Differenza"] = (
    confronto["Valore attuale"]
    - confronto["Valore iniziale"]
)

st.dataframe(
    confronto.style.format({
        "Valore iniziale": "{:.0f}",
        "Valore attuale": "{:.0f}",
        "Differenza": "{:+.0f}"
    }),
    use_container_width=True,
    hide_index=True
)
