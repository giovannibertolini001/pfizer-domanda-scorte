Project Work – Previsione della domanda e gestione delle scorte

Obiettivo del progetto:

Il progetto risponde alla traccia «Sviluppo di una mini-dashboard Python per la previsione della domanda e la pianificazione delle scorte di un'impresa».

È stata realizzata un'applicazione in Python, utilizzando la libreria Streamlit, per simulare e analizzare la domanda di prodotti farmaceutici e supportare la gestione delle scorte.

Come azienda di riferimento è stata scelta Pfizer.

Prodotti inseriti:

Il programma prende in considerazione sei prodotti farmaceutici:

- Zithromax: antibiotico utilizzato per il trattamento di diverse infezioni batteriche, soprattutto delle vie respiratorie.

- Vfend: farmaco antifungino impiegato nel trattamento di infezioni causate da funghi, anche di tipo grave.

- Paxlovid: farmaco antivirale utilizzato per il trattamento del COVID-19 in pazienti a rischio di sviluppare forme gravi della malattia.

- Prevenar 20: vaccino pneumococcico utilizzato per la prevenzione di malattie causate da determinati tipi di batterio pneumococco.

- Abrysvo: vaccino utilizzato per la prevenzione delle malattie delle basse vie respiratorie causate dal virus respiratorio sinciziale (RSV).

- Eliquis: farmaco anticoagulante utilizzato per prevenire e trattare la formazione di coaguli di sangue.

Cosa fa il programma:

Il programma permette di:

- Generare e visualizzare la domanda storica simulata dei prodotti.
- Prevedere la domanda futura attraverso tre metodi: Media Mobile, Smorzamento Esponenziale e Holt-Winters.
- Confrontare i risultati dei metodi previsionali e individuare quello più accurato.
- Calcolare le scorte di sicurezza, cicliche, anticipate e in transito.
- Determinare il punto di riordino dei prodotti.
- Modificare i parametri della domanda e dell'approvvigionamento.
- Confrontare diversi scenari di gestione delle scorte.
- Visualizzare i risultati attraverso grafici e tabelle.

Avvio del programma

Installare le librerie necessarie:

pip install -r requirements.txt

Avviare la dashboard:

streamlit run "programma project work.py"
## Nota

Il progetto è stato sviluppato a scopo universitario. I dati utilizzati sono simulati e non rappresentano dati aziendali reali di Pfizer.
