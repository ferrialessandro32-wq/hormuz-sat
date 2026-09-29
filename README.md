# Radar satellitare per il Registro Hormuz

Conta le grandi navi (≥180 m e ≥250 m) dalle immagini radar
Copernicus Sentinel-1, che vedono anche le petroliere con il transponder AIS spento. Gira da solo su
GitHub ogni 6 ore e pubblica `data/latest.json`, che l'agente Registro Hormuz legge ogni sera.

Dati: Copernicus Sentinel-1, accesso libero e gratuito anche per uso commerciale.

## Zone controllate

| Zona | Cosa misura |
|---|---|
| `hormuz` | corsie di traffico dello stretto |
| `yanbu`, `fujairah` | terminal degli oleodotti di bypass |
| `sohar_sts` | area di trasbordi ship-to-ship nel Golfo di Oman |
| `ras_tanura`, `mina_ahmadi`, `basra`, `kharg`, `das` | terminal di carico nel Golfo: quanto greggio viene caricato |

Per ogni immagine: navi stimate ≥180 m e ≥250 m, coppie affiancate in trasbordo (`n_sts`, contano come 2 navi),
oggetti di più navi in fila (`n_merged`, contati lunghezza/300 m).

## Attivazione (una volta sola, circa 15 minuti)

1. **Account Copernicus** (gratuito): registrati su https://dataspace.copernicus.eu.
   Poi vai su https://shapps.dataspace.copernicus.eu/dashboard → *User settings* → *OAuth clients* →
   *Create*, scegli "Never expire" e copia subito **client ID** e **client secret** (il secret non si
   rivede più).
2. **Repository GitHub pubblico**: crea un nuovo repository (es. `hormuz-sat`) e carica questi file:
   `sat_ships.py`, `requirements.txt`, `.github/workflows/sat.yml`, `README.md`.
   Il repository deve essere pubblico perché l'agente legga il file dei conteggi; contiene solo
   numeri di navi, nessun dato personale.
3. **Segreti**: nel repository → *Settings* → *Secrets and variables* → *Actions* → *New repository secret*:
   - `CDSE_CLIENT_ID` = client ID del passo 1
   - `CDSE_CLIENT_SECRET` = client secret del passo 1
4. **Prima esecuzione**: *Actions* → "Conteggio navi da satellite" → *Run workflow*. Dopo qualche minuto
   compare `data/latest.json`.
5. Comunica a Claude l'indirizzo del repository: imposterà nell'archivio del Registro Hormuz l'indirizzo
   `https://raw.githubusercontent.com/<utente>/hormuz-sat/main/data/latest.json`.

## Cosa misura e cosa no

- È un **conteggio istantaneo** delle navi lunghe presenti nella zona al passaggio del satellite, non un
  flusso in barili. Serve come indice indipendente: se Windward vede 5 petroliere e il radar ne vede 12
  all'ancora, molte viaggiano al buio.
- Con Sentinel-1C e 1D la stessa zona viene ripresa ogni 2–6 giorni: non c'è un'immagine ogni giorno.
- Il rilevamento è testato su immagini simulate (lunghezze entro il 4%, nessun falso positivo). Su
  immagini reali vento forte, piattaforme e scie possono creare errori: va calibrato per una settimana
  confrontandolo con Windward.
- Le navi a meno di ~250 m dalla costa o dai moli non vengono contate.
