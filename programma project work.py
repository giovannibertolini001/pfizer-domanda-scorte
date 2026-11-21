import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    ExponentialSmoothing
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# CONFIGURAZIONE

st.set_page_config(
    page_title="Pfizer - Domanda e Scorte",
    page_icon="💊",
    layout="wide"
)


# PRODOTTI


prodotti = {
    "Zithromax": {
        "media": 8000,
        "variabilita": 0.08,
        "stagionalita": 0.40,
        "settimana_picco": 3,
        "trend": 0.00,
        "lead_time": 2,
        "servizio": 0.95,
        "quantita_ordine": 16000
    },
    "Vfend": {
        "media": 2500,
        "variabilita": 0.10,
        "stagionalita": 0.08,
        "settimana_picco": 3,
        "trend": 0.02,
        "lead_time": 3,
        "servizio": 0.95,
        "quantita_ordine": 7500
    },
    "Paxlovid": {
        "media": 3500,
        "variabilita": 0.25,
        "stagionalita": 0.30,
        "settimana_picco": 2,
        "trend": 0.00,
        "lead_time": 1,
        "servizio": 0.95,
        "quantita_ordine": 7000
    },
    "Prevenar 20": {
        "media": 4500,
        "variabilita": 0.08,
        "stagionalita": 0.45,
        "settimana_picco": 44,
        "trend": 0.02,
        "lead_time": 4,
        "servizio": 0.98,
        "quantita_ordine": 18000
    },
    "Abrysvo": {
        "media": 3000,
        "variabilita": 0.08,
        "stagionalita": 0.55,
        "settimana_picco": 44,
        "trend": 0.05,
        "lead_time": 3,
        "servizio": 0.98,
        "quantita_ordine": 12000
    },
    "Eliquis": {
        "media": 12000,
        "variabilita": 0.04,
        "stagionalita": 0.03,
        "settimana_picco": 3,
        "trend": 0.08,
        "lead_time": 2,
        "servizio": 0.95,
        "quantita_ordine": 24000
    }
}



# Generazione della domanda

def genera_domanda(media, variabilita, stagionalita, trend,
                   settimana_picco=0, settimane=156, picchi=False, seed=None):

    rng = np.random.default_rng(seed)
    t = np.arange(settimane)

    # componenti della domanda
    componente_trend = media * trend * (t / 52)
    componente_stagionale = media * stagionalita * np.cos(2 * np.pi * (t - settimana_picco) / 52)
    componente_casuale = rng.normal(0, media * variabilita, settimane)
    domanda = media + componente_trend + componente_stagionale + componente_casuale

    # per Paxlovid vengono simulati alcuni picchi di domanda
    if picchi:
        settimane_picco = rng.choice(settimane, size=4, replace=False)

        for settimana in settimane_picco:
            domanda[settimana] *= rng.uniform(1.5, 2.2)

    return np.maximum(domanda, 0)


# Media mobile

@st.cache_data
def previsione_media_mobile(train, numero_periodi=12, finestra=4, test=None):
    storico = list(train)
    previsioni = []

    for i in range(numero_periodi):
        valore = np.mean(storico[-finestra:])
        previsioni.append(valore)

        if test is not None:
            storico.append(test.iloc[i] if hasattr(test, "iloc") else test[i])
        else:
            storico.append(valore)

    return np.array(previsioni)


# Exponential Smoothing

@st.cache_data
def previsione_exponential_smoothing(train, numero_periodi=12, test=None):
    if test is None:
        modello = SimpleExpSmoothing(
            train,
            initialization_method="estimated"
        ).fit(optimized=True)

        return np.maximum(
            np.array(modello.forecast(numero_periodi)),
            0
        )

    storico = list(train)
    previsioni = []

    for i in range(numero_periodi):

        modello = SimpleExpSmoothing(
            storico,
            initialization_method="estimated"
        ).fit(optimized=True)

        
        valore = float(modello.forecast(1)[0])
        previsioni.append(valore)

        storico.append(test.iloc[i] if hasattr(test, "iloc") else test[i])

    return np.array(previsioni)



# Holt-Winters

@st.cache_data
def previsione_holt_winters(train, numero_periodi=12, test=None):
    if test is None:
        modello = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=52
        ).fit(optimized=True)

        return np.maximum(
            np.array(modello.forecast(numero_periodi)),
            0
        )

    storico = list(train)
    previsioni = []

    for i in range(numero_periodi):

        modello = ExponentialSmoothing(
            storico,
            trend="add",
            seasonal="add",
            seasonal_periods=52
        ).fit(optimized=True)

        valore = max(float(modello.forecast(1)[0]), 0)
        previsioni.append(valore)

        
        storico.append(test.iloc[i] if hasattr(test, "iloc") else test[i])

    return np.array(previsioni)

# Calcolo degli errori

def calcola_metriche(reale, previsto):

    mae = mean_absolute_error(reale, previsto)
    rmse = np.sqrt(mean_squared_error(reale, previsto))

    reale = np.array(reale)
    previsto = np.array(previsto)

    valori_validi = reale != 0

    mape = np.mean(
        np.abs((reale[valori_validi] - previsto[valori_validi]) / reale[valori_validi])
    ) * 100

    return mae, rmse, mape

@st.cache_data
def calcola_scorta_anticipata(serie_completa):
    modello = ExponentialSmoothing(
        serie_completa,
        trend="add",
        seasonal="add",
        seasonal_periods=52
    ).fit(optimized=True)

    profilo = np.maximum(
        np.array(modello.forecast(52)),
        0
    )

    accumulo = np.cumsum(profilo - profilo.mean())

    return float(max(accumulo.max() - accumulo.min(), 0))

# Calcolo delle scorte

def calcola_scorte(domanda_media, deviazione_standard, lead_time,
                   livello_servizio, quantita_ordine, scorta_anticipata=0):

    valori_z = {
        0.90: 1.28,
        0.95: 1.65,
        0.98: 2.05,
        0.99: 2.33
    }

    z = valori_z[livello_servizio]

    safety_stock = z * deviazione_standard * np.sqrt(lead_time)
    cycle_stock = quantita_ordine / 2
    # Scorta anticipata stimata separatamente sul profilo
    # della domanda prevista per le prossime 52 settimane
    anticipated_stock = scorta_anticipata
    pipeline_stock = domanda_media * lead_time

    average_stock = cycle_stock + safety_stock + anticipated_stock
    reorder_point = domanda_media * lead_time + safety_stock

    return {
        "Safety Stock": safety_stock,
        "Cycle Stock": cycle_stock,
        "Anticipated Stock": anticipated_stock,
        "Pipeline Stock": pipeline_stock,
        "Average Stock": average_stock,
        "Reorder Point": reorder_point
    }


# Titolo

st.title("Pfizer - Analisi della domanda e delle scorte")
with st.expander("Descrizione dell'impresa e contesto logistico"):
    st.write("""
Pfizer è un'impresa multinazionale che opera nel settore
farmaceutico, occupandosi dello sviluppo, della produzione
e della commercializzazione di medicinali e vaccini.

    

    
Il progetto analizza sei prodotti farmaceutici:

- Zithromax: antibiotico.
- Vfend: antifungino.
- Paxlovid: antivirale.
- Prevenar 20: vaccino pneumococcico.
- Abrysvo: vaccino contro il virus respiratorio sinciziale (RSV).
- Eliquis: anticoagulante.

I prodotti appartengono a categorie terapeutiche differenti
e sono utilizzati per simulare diversi andamenti della domanda.


La gestione delle scorte deve considerare la variabilità
della domanda, i tempi di approvvigionamento e le
condizioni di conservazione dei prodotti.

Alcuni vaccini richiedono il mantenimento della catena
del freddo. Inoltre, la scadenza dei medicinali rende
importante evitare accumuli eccessivi di magazzino.

Le serie storiche utilizzate nel progetto sono simulate
a scopo didattico e non rappresentano dati commerciali
effettivi di Pfizer.
""")

st.write(
 "Il programma permette di analizzare la domanda dei prodotti, "
 "confrontare diversi metodi di previsione e calcolare le scorte."
)

st.caption(
"I dati utilizzati sono simulati a fini didattici e non rappresentano dati reali di vendita."
)

## Selezione del prodotto

prodotto = st.selectbox("Seleziona il prodotto", list(prodotti.keys()))
parametri = prodotti[prodotto]

# Parametri

st.subheader("Parametri della domanda e dell'approvvigionamento")

col1, col2, col3 = st.columns(3)

with col1:
    media = st.number_input(
        "Domanda media settimanale (confezioni/settimana)",
        min_value=100,
        value=parametri["media"],
        step=100,
        key=f"media_{prodotto}"
    )

    variabilita = st.slider(
        "Variabilità", 0.0, 0.50,
        float(parametri["variabilita"]), 0.01,
        key=f"variabilita_{prodotto}"
    )

    stagionalita = st.slider(
        "Stagionalità", 0.0, 0.60,
        float(parametri["stagionalita"]), 0.01,
        key=f"stagionalita_{prodotto}"
    )

with col2:
    trend = st.slider(
        "Trend annuo", -0.10, 0.10,
        float(parametri["trend"]), 0.01,
        key=f"trend_{prodotto}"
    )

    lead_time = st.number_input(
        "Tempo di approvvigionamento (settimane)",
        min_value=1,
        max_value=12,
        value=parametri["lead_time"],
        key=f"lead_time_{prodotto}"
    )

with col3:
    servizio = st.selectbox(
        "Livello di servizio",
        [0.90, 0.95, 0.98, 0.99],
        index=[0.90, 0.95, 0.98, 0.99].index(parametri["servizio"]),
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


# Generazione della domanda

if "seed" not in st.session_state:
    st.session_state.seed = 6

if st.button("🔄 Genera nuova domanda"):
    st.session_state.seed += 1

domanda = genera_domanda(
    media,
    variabilita,
    stagionalita,
    trend,
    settimana_picco=parametri["settimana_picco"],
    settimane=156,
    picchi=(prodotto == "Paxlovid"),
    seed=st.session_state.seed
)


# Serie temporale

serie = pd.Series(domanda)

train = serie.iloc[:144]
test = serie.iloc[144:]

date = pd.date_range("2023-01-02", periods=156, freq="W-MON")
date_test = date[-12:]


# PREVISIONI

previsione_mm = previsione_media_mobile(train, test=test)
previsione_es = previsione_exponential_smoothing(train, test=test)
previsione_hw = previsione_holt_winters(train, test=test)


# Metriche

metriche_mm = calcola_metriche(test, previsione_mm)
metriche_es = calcola_metriche(test, previsione_es)
metriche_hw = calcola_metriche(test, previsione_hw)

metodi = {
    "Media mobile": metriche_mm[2],
    "Previsione esponenziale": metriche_es[2],
    "Holt-Winters": metriche_hw[2]
}

previsioni = {
    "Media mobile": previsione_mm,
    "Previsione esponenziale": previsione_es,
    "Holt-Winters": previsione_hw
}

metodo_migliore = min(metodi, key=metodi.get)
previsione_migliore = previsioni[metodo_migliore]
# Previsione delle prossime 12 settimane
numero_settimane_future = 12

if metodo_migliore == "Media mobile":
    previsione_futura = previsione_media_mobile(
        serie,
        numero_periodi=numero_settimane_future
    )

elif metodo_migliore == "Previsione esponenziale":
    previsione_futura = previsione_exponential_smoothing(
        serie,
        numero_periodi=numero_settimane_future
    )

else:
    previsione_futura = previsione_holt_winters(
        serie,
        numero_periodi=numero_settimane_future
    )

date_future = pd.date_range(
    start=date[-1] + pd.Timedelta(weeks=1),
    periods=numero_settimane_future,
    freq="W-MON"
)

# Andamento storico della domanda

st.subheader("Andamento storico della domanda")


st.write("Seleziona il periodo da visualizzare")

col_data1, col_data2 = st.columns(2)

with col_data1:
    data_inizio = st.date_input(
        "Data iniziale",
        value=date[0].date(),
        min_value=date[0].date(),
        max_value=date_future[-1].date()
    )

with col_data2:
    data_fine = st.date_input(
        "Data finale",
        value=date_future[-1].date(),
        min_value=date[0].date(),
        max_value=date_future[-1].date()
    )

date_valide = data_inizio <= data_fine

if not date_valide:
    st.warning("La data iniziale deve precedere quella finale.")

if date_valide:
    fig_storico, ax_storico = plt.subplots(figsize=(6, 3.5))
    ax_storico.set_xlim(
        pd.Timestamp(data_inizio),
        pd.Timestamp(data_fine) + pd.Timedelta(days=1)
    )


    ax_storico.plot(
        date,
        serie.values,
        label="Domanda storica",
        linewidth=1.5
    )

    ax_storico.plot(
        pd.DatetimeIndex([date[-1]]).append(date_future),
        np.concatenate(([serie.iloc[-1]], previsione_futura)),
        label="Domanda prevista (12 settimane)",
        color="orange",
        linestyle="--",
        linewidth=2
    )

    ax_storico.legend()


    date_inizio_grafico = pd.Timestamp(data_inizio)
    date_fine_grafico = pd.Timestamp(data_fine)

    date_intermedie = pd.date_range(
        start=date_inizio_grafico,
        end=date_fine_grafico,
        periods=5
    )


    date_asse = date_intermedie.unique()


    ax_storico.set_xticks(date_asse)
    ax_storico.xaxis.set_major_formatter(
        mdates.DateFormatter("%d/%m/%Y")
    )

    ax_storico.set_xlabel("Data")
    ax_storico.set_ylabel("Confezioni/settimana")
    ax_storico.grid(True, alpha=0.3)

    fig_storico.autofmt_xdate(rotation=30)
    fig_storico.tight_layout()

    st.pyplot(fig_storico, width="content")
    plt.close(fig_storico)



# Grafici delle previsioni

inizio_x = date_test.min()
fine_x = date_test.max() + pd.Timedelta(days=1)


col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Media mobile")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(date_test, test.values, label="Domanda reale")
    
    
    ax1.plot(
        date_test,
        previsione_mm,
        label="Previsione media mobile"
    )
    ax1.set_xlim(inizio_x, fine_x)
    
    date_asse_mm = pd.date_range(
        start=inizio_x,
        end=fine_x - pd.Timedelta(days=1),
        periods=5
    )

    ax1.set_xticks(date_asse_mm)
    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter("%d/%m/%Y")
    )

    ax1.set_xlabel("Data")
    ax1.set_ylabel("Confezioni/settimana")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig1.autofmt_xdate(rotation=30)
    fig1.tight_layout()

    st.pyplot(fig1)
    plt.close(fig1)


with col2:
    st.subheader("2. Previsione esponenziale")

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.plot(date_test, test.values, label="Domanda reale")
    ax2.plot(date_test, previsione_es, label="Previsione esponenziale")
    ax2.set_xlim(inizio_x, fine_x)
    
    date_asse_es = pd.date_range(
        start=inizio_x,
        end=fine_x - pd.Timedelta(days=1),
        periods=5
    )

    ax2.set_xticks(date_asse_es)
    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter("%d/%m/%Y")
    )


    ax2.set_xlabel("Data")
    ax2.set_ylabel("Confezioni/settimana")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.autofmt_xdate(rotation=30)
    fig2.tight_layout()

    st.pyplot(fig2)
    plt.close(fig2)


col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Holt-Winters")

    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.plot(date_test, test.values, label="Domanda reale")
    ax3.plot(date_test, previsione_hw, label="Holt-Winters")
    ax3.set_xlim(inizio_x, fine_x)
    
    date_asse_hw = pd.date_range(
        start=inizio_x,
        end=fine_x - pd.Timedelta(days=1),
        periods=5
    )

    ax3.set_xticks(date_asse_hw)
    ax3.xaxis.set_major_formatter(
        mdates.DateFormatter("%d/%m/%Y")
    )


    ax3.set_xlabel("Data")
    ax3.set_ylabel("Confezioni/settimana")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig3.autofmt_xdate(rotation=30)
    fig3.tight_layout()

    st.pyplot(fig3)
    plt.close(fig3)

with col4:
    st.subheader("4. Accuratezza dei metodi (MAPE)")

    fig4, ax4 = plt.subplots(figsize=(5, 3))

    ax4.bar(
        list(metodi.keys()),
        list(metodi.values())
    )

    ax4.set_ylabel("MAPE (%)")
    ax4.tick_params(axis="x", labelrotation=15)
    ax4.grid(True, axis="y", alpha=0.3)

    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

# Calcolo delle scorte iniziali e attuali

# Calcolo delle scorte dello scenario iniziale

domanda_base = genera_domanda(
    parametri["media"],
    parametri["variabilita"],
    parametri["stagionalita"],
    parametri["trend"],
    settimana_picco=parametri["settimana_picco"],
    settimane=156,
    picchi=(prodotto == "Paxlovid"),
    seed=st.session_state.seed
)

train_base = domanda_base[:144]
test_base = domanda_base[144:]


previsioni_base = {
    "Media mobile": previsione_media_mobile(train_base, test=test_base),
    "Previsione esponenziale": previsione_exponential_smoothing(train_base, test=test_base),
    "Holt-Winters": previsione_holt_winters(train_base, test=test_base)
}
errori_base = {
    nome: calcola_metriche(test_base, valori)[2]
    for nome, valori in previsioni_base.items()
}

metodo_base = min(errori_base, key=errori_base.get)
if metodo_base == "Media mobile":
    previsione_futura_base = previsione_media_mobile(
        domanda_base, numero_periodi=12
    )
elif metodo_base == "Previsione esponenziale":
    previsione_futura_base = previsione_exponential_smoothing(
        domanda_base, numero_periodi=12
    )
else:
    previsione_futura_base = previsione_holt_winters(
        domanda_base, numero_periodi=12
    )

domanda_media_base = np.mean(previsione_futura_base)

scorte_base = calcola_scorte(
    domanda_media_base,
    np.std(
        test_base - previsioni_base[metodo_base],
        ddof=1
    ),
    parametri["lead_time"],
    parametri["servizio"],
    parametri["quantita_ordine"],
    scorta_anticipata=calcola_scorta_anticipata(domanda_base)
)

domanda_media_prevista = np.mean(previsione_futura)
deviazione_attuale = np.std(
    test.values - previsione_migliore,
    ddof=1
)


scorte_attuali = calcola_scorte(
    domanda_media_prevista,
    deviazione_attuale,
    lead_time,
    servizio,
    quantita_ordine,
    scorta_anticipata=calcola_scorta_anticipata(serie)
)

# Visualizzazione delle scorte

st.subheader("Gestione delle scorte")

# Parametri economici per la simulazione
costo_unitario = st.number_input(
    "Costo unitario ipotetico per confezione (€)",
    min_value=0.0,
    value=20.0,
    step=1.0
)

tasso_mantenimento = st.number_input(
    "Tasso annuo di mantenimento delle scorte (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=1.0
)

# Calcolo del costo annuo di mantenimento
costo_mantenimento = (
    scorte_attuali["Average Stock"]
    * costo_unitario
    * (tasso_mantenimento / 100)
)

st.metric(
    "Costo annuo stimato di mantenimento",
    f"{costo_mantenimento:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €"
)

# Indicatore teorico del rischio di stockout
rischio_stockout = (1 - servizio) * 100

st.metric(
    "Rischio teorico di stockout per ciclo (%)",
    f"{rischio_stockout:.1f}%"
)

st.caption(
    "Indicatore teorico derivato dal livello di servizio scelto. "
    "Non rappresenta una probabilità di stockout misurata sui dati reali."
)
# Rischio di scadenza e obsolescenza
if prodotto in ["Prevenar 20", "Abrysvo"]:
    st.warning(
        "Per questo vaccino è importante considerare "
        "la catena del freddo e la data di scadenza. "
        "Scorte eccessive possono aumentare il rischio "
        "di sprechi e obsolescenza."
    )
else:
    st.info(
        "Anche per questo medicinale è importante "
        "controllare le date di scadenza. "
        "Scorte eccessive possono causare sprechi "
        "e maggiori costi di magazzino."
    )



col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Scorta di sicurezza",
        f"{scorte_attuali['Safety Stock']:.0f}"
    )
    st.caption(
        "Calcolata usando la deviazione standard degli errori "
        "delle ultime 12 settimane, il lead time e il livello "
        "di servizio. La stima è indicativa."
    )

with col2:
    st.metric(
        "Scorta ciclica",
        f"{scorte_attuali['Cycle Stock']:.0f}"
    )


with col3:
    st.metric(
        "Scorta anticipata",
        f"{scorte_attuali['Anticipated Stock']:.0f}"
    )
    st.caption(
    "Scorta anticipata stimata con Holt-Winters su 52 settimane, "
    "considerando le variazioni stagionali della domanda "
    "e ipotizzando un approvvigionamento costante."
    )




with col4:
    st.metric(
        "Quantità in arrivo",
        f"{scorte_attuali['Pipeline Stock']:.0f}"
    )
    st.caption(
        "Quantità teorica in transito, stimata moltiplicando "
        "la domanda media settimanale prevista per il tempo "
        "di approvvigionamento. Non rappresenta ordini reali."
    )


st.write(
    f"Scorta media: {scorte_attuali['Average Stock']:.0f} | "
    f"Punto di riordino: {scorte_attuali['Reorder Point']:.0f}"
)

st.caption(
    "La scorta media comprende le scorte cicliche, di sicurezza "
    "e anticipate. La scorta pipeline non viene inclusa perché "
    "rappresenta merce teoricamente in transito, non ancora "
    "disponibile nel magazzino."
)


# Accuratezza delle previsioni

st.subheader("Confronto degli errori di previsione")
st.success(
    f"Per {prodotto}, il metodo con l'errore più basso è "
    f"{metodo_migliore} (MAPE: {metodi[metodo_migliore]:.2f}%)"
)
st.caption(
    "I metodi sono confrontati sulle ultime 12 settimane "
    "mediante previsioni a un passo, aggiornate ogni settimana "
    "con la domanda osservata. Il metodo con MAPE minore "
    "viene successivamente utilizzato per prevedere "
    "le 12 settimane future. L'accuratezza a un passo "
    "non garantisce la stessa accuratezza su 12 settimane."
)

risultati = pd.DataFrame({
    "Metodo": ["Media mobile", "Previsione esponenziale", "Holt-Winters"],
    "MAE": [metriche_mm[0], metriche_es[0], metriche_hw[0]],
    "RMSE": [metriche_mm[1], metriche_es[1], metriche_hw[1]],
    "MAPE (%)": [metriche_mm[2], metriche_es[2], metriche_hw[2]]
})

st.dataframe(
    risultati.round({"MAE": 0, "RMSE": 0, "MAPE (%)": 2}),
    width="stretch",
    hide_index=True
)
st.caption(
    "Le serie storiche sono generate con caratteristiche "
    "differenti di trend, stagionalità e variabilità. "
    "La componente stagionale simulata ha un periodo "
    "di 52 settimane, lo stesso utilizzato dal modello "
    "Holt-Winters. Questo può favorire il metodo "
    "nei prodotti con forte stagionalità. "
    "Nei prodotti con stagionalità ridotta, invece, "
    "anche i metodi più semplici possono ottenere "
    "risultati competitivi. La scelta del metodo migliore "
    "dipende dal MAPE calcolato sui dati di verifica."
)

# Confronto prima e dopo

st.subheader("Confronto tra situazione iniziale e modificata")
indicatori = {
    "Scorta di sicurezza": "Safety Stock",
    "Scorta ciclica": "Cycle Stock",
    "Scorta anticipata": "Anticipated Stock",
    "Quantità in arrivo": "Pipeline Stock",
    "Scorta media": "Average Stock",
    "Punto di riordino": "Reorder Point"
}

confronto = pd.DataFrame({
    "Indicatore": list(indicatori.keys()),
    "Parametri di default": [scorte_base[x] for x in indicatori.values()],
    "Parametri modificati": [scorte_attuali[x] for x in indicatori.values()]
})

confronto["Differenza"] = (
    confronto["Parametri modificati"] -
    confronto["Parametri di default"]
)

st.dataframe(
    confronto.round(0),
    width="stretch",
    hide_index=True
)
st.subheader("Composizione delle scorte nei due scenari")

tipi = [
    "Safety Stock",
    "Cycle Stock",
    "Anticipated Stock",
    "Pipeline Stock"
]

nomi = [
    "Sicurezza",
    "Ciclica",
    "Anticipata",
    "Pipeline (in transito)"
]

scenari = ["Parametri di default", "Parametri modificati"]

fig5, ax5 = plt.subplots(figsize=(6, 3))

base = np.zeros(2)

for tipo, nome in zip(tipi, nomi):
    valori = np.array([
        scorte_base[tipo],
        scorte_attuali[tipo]
    ])

    ax5.bar(scenari, valori, bottom=base, label=nome)
    base += valori
ax5.set_ylabel("Confezioni")
ax5.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax5.grid(True, axis="y", alpha=0.3)

fig5.tight_layout()
st.pyplot(fig5)
plt.close(fig5)
# Esportazione delle serie storiche simulate
st.subheader("Esportazione dei dati")

righe = []


for nome, p in prodotti.items():
    serie_prodotto = genera_domanda(
        p["media"],
        p["variabilita"],
        p["stagionalita"],
        p["trend"],
        settimana_picco=p["settimana_picco"],
        settimane=156,
        picchi=(nome == "Paxlovid"),
        seed=6
    )

    righe.append(pd.DataFrame({
        "data": date.date,
        "prodotto": nome,
        "domanda": serie_prodotto.round(0).astype(int)
    }))


dati_export = pd.concat(righe, ignore_index=True)

st.download_button(
    "📥 Scarica le serie storiche (CSV)",
    data=dati_export.to_csv(index=False, sep=";").encode("utf-8"),
    file_name="serie_domanda_pfizer.csv",
    mime="text/csv"
)

