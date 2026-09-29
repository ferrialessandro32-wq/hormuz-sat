"""Conteggio giornaliero delle grandi navi da immagini radar Sentinel-1 (Copernicus).

Per tre zone (Stretto di Hormuz, Yanbu, Fujairah) cerca le nuove acquisizioni
Sentinel-1 GRD, scarica il backscatter VV a 20 m tramite la Process API del
Copernicus Data Space Ecosystem, individua le navi come punti brillanti su mare
scuro (CFAR) e ne stima la lunghezza. Il radar vede anche le navi con il
transponder AIS spento.

Uscite (cartella data/):
  sat_counts.csv  una riga per zona e per acquisizione
  latest.json     ultima acquisizione per zona, letta dall'agente Registro Hormuz

Credenziali (variabili d'ambiente): CDSE_CLIENT_ID, CDSE_CLIENT_SECRET.
Dati: Copernicus Sentinel data, accesso libero, pieno e gratuito anche per uso commerciale.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

PIXEL_M = 20.0  # risoluzione richiesta in metri

# bbox: lon_min, lat_min, lon_max, lat_max. Tenute sotto i 2500 px per lato a 20 m.
AOIS = {
    # corsie di traffico dello stretto (schema di separazione del traffico)
    "hormuz": (56.20, 26.30, 56.62, 26.70),
    # terminal e ancoraggio di Yanbu (Mar Rosso)
    "yanbu": (37.90, 23.85, 38.25, 24.15),
    # ancoraggio e terminal di Fujairah (Golfo di Oman)
    "fujairah": (56.33, 25.00, 56.62, 25.35),
    # area di trasbordi ship-to-ship al largo di Sohar (Golfo di Oman)
    "sohar_sts": (56.60, 24.30, 57.00, 24.65),
    # terminal di carico dentro il Golfo: dicono quanto greggio viene caricato
    "ras_tanura": (50.00, 26.60, 50.30, 27.00),   # Ras Tanura e boe di Juaymah (Arabia Saudita)
    "mina_ahmadi": (48.05, 28.95, 48.35, 29.20),  # Mina al-Ahmadi (Kuwait)
    "basra": (48.65, 29.55, 48.95, 29.90),        # terminal offshore di Basra e Khor al-Amaya (Iraq)
    "kharg": (50.20, 29.10, 50.45, 29.35),        # isola di Kharg (Iran)
    "das": (52.80, 24.82, 53.12, 25.22),          # isole di Das e Zirku (Emirati)
}

# soglie di lunghezza: >=180 m include Aframax e più grandi; >=250 m Suezmax e VLCC
LEN_BIG = 180.0
LEN_VLCC = 250.0

LOOKBACK_DAYS = 4
DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "sat_counts.csv"
LATEST_PATH = DATA_DIR / "latest.json"
CSV_FIELDS = [
    "acq_utc", "date_utc", "aoi", "scene_id", "n_ge180m", "n_ge250m",
    "n_sts", "n_merged", "n_objects", "lengths_m", "sea_fraction", "processed_utc",
]
# Oggetti larghi almeno STS_MIN_WIDTH e lunghi almeno STS_MIN_LEN: due navi affiancate (trasbordo).
STS_MIN_WIDTH = 100.0
STS_MIN_LEN = 230.0
# Oggetti più lunghi di MERGED_LEN: più navi in fila (o nave con scia), contate come lunghezza/300 m.
MERGED_LEN = 400.0

EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "dataMask"]}],
          output: {bands: 2, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.VV, s.dataMask]; }
"""


# ---------------------------------------------------------------- rete

def get_token(session) -> str:
    cid = os.environ.get("CDSE_CLIENT_ID")
    secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit("Mancano CDSE_CLIENT_ID o CDSE_CLIENT_SECRET.")
    r = session.post(TOKEN_URL, data={
        "grant_type": "client_credentials", "client_id": cid, "client_secret": secret,
    }, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def search_scenes(session, token, bbox, start, end):
    body = {
        "collections": ["sentinel-1-grd"],
        "bbox": list(bbox),
        "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
        "limit": 50,
    }
    r = session.post(CATALOG_URL, json=body,
                     headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    out = []
    for f in r.json().get("features", []):
        props = f.get("properties", {})
        out.append({"id": f.get("id"), "datetime": props.get("datetime")})
    return out


def fetch_vv(session, token, bbox, acq_iso):
    """Scarica VV (sigma0 lineare) e dataMask per l'acquisizione indicata."""
    t = datetime.fromisoformat(acq_iso.replace("Z", "+00:00"))
    width, height = bbox_pixels(bbox)
    body = {
        "input": {
            "bounds": {"bbox": list(bbox),
                       "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": f"{t - timedelta(minutes=2):%Y-%m-%dT%H:%M:%SZ}",
                                  "to": f"{t + timedelta(minutes=2):%Y-%m-%dT%H:%M:%SZ}"},
                    "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH",
                },
                "processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True,
                               "demInstance": "COPERNICUS_30"},
            }],
        },
        "output": {"width": width, "height": height,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    r = session.post(PROCESS_URL, json=body, headers={
        "Authorization": f"Bearer {token}", "Accept": "image/tiff"}, timeout=300)
    r.raise_for_status()
    import tifffile
    arr = tifffile.imread(io.BytesIO(r.content))
    if arr.ndim == 3 and arr.shape[0] == 2 and arr.shape[-1] != 2:
        arr = np.moveaxis(arr, 0, -1)
    return arr[..., 0].astype(np.float32), arr[..., 1] > 0


def bbox_pixels(bbox):
    lon0, lat0, lon1, lat1 = bbox
    lat_mid = np.radians((lat0 + lat1) / 2)
    w_m = (lon1 - lon0) * 111_320 * np.cos(lat_mid)
    h_m = (lat1 - lat0) * 110_574
    return int(round(w_m / PIXEL_M)), int(round(h_m / PIXEL_M))


# ---------------------------------------------------------------- rilevamento

def detect_ships(vv: np.ndarray, valid: np.ndarray, pixel_m: float = PIXEL_M):
    """Restituisce (lunghezze in metri, frazione di mare) delle navi individuate.

    Metodo: mare = pixel scuri (sigma0 VV basso) in un intorno ampio; nave = pixel
    molto più brillanti del proprio intorno di mare (CFAR), raggruppati in oggetti.
    Oggetti enormi (terra, piattaforme, porti) vengono scartati.
    """
    vv = np.where(valid, vv, np.nan)
    db = 10 * np.log10(np.clip(vv, 1e-6, None))

    # Fondo locale = mediana su finestra ampia, calcolata su una griglia ridotta di 4x
    # (veloce e non falsata dalle navi, che occupano pochi pixel).
    f = 4
    h, w = db.shape
    hc, wc = h // f * f, w // f * f
    coarse = np.nanmedian(
        np.nan_to_num(db[:hc, :wc], nan=0.0).reshape(hc // f, f, wc // f, f), axis=(1, 3))
    coarse = ndimage.median_filter(coarse, size=11)          # ~ 44 px = 880 m
    bg = np.full(db.shape, 0.0, dtype=np.float32)
    bg[:hc, :wc] = np.kron(coarse, np.ones((f, f), dtype=np.float32))
    bg[hc:, :] = bg[hc - 1:hc, :] if hc else 0.0
    bg[:, wc:] = bg[:, wc - 1:wc] if wc else 0.0

    # Mare = fondo scuro (mare aperto -18/-25 dB), lontano almeno ~250 m da terra o bordo immagine.
    sea_raw = (bg < -14.0) & valid
    sea = ndimage.distance_transform_edt(sea_raw) * pixel_m >= 250.0
    sea_fraction = float(sea.mean()) if sea.size else 0.0
    if sea.sum() < 1000:
        return [], sea_fraction

    # Nave = almeno 10 dB sopra il fondo di mare e sopra -10 dB in assoluto.
    cand = sea & (db > bg + 10.0) & (db > -10.0)
    cand = ndimage.binary_closing(cand, structure=np.ones((3, 3)), iterations=1)
    labels, n = ndimage.label(cand)
    if n == 0:
        return [], sea_fraction

    objs = []
    for sl, idx in zip(ndimage.find_objects(labels), range(1, n + 1)):
        ys, xs = np.nonzero(labels[sl] == idx)
        npx = ys.size
        if npx < 4:            # troppo piccolo per una nave grande
            continue
        if npx > 1500:         # troppo grande: terra, piattaforma, molo
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        pts -= pts.mean(axis=0)
        # asse principale, poi estensione reale lungo l'asse (max - min) + 1 pixel
        _, vecs = np.linalg.eigh(np.cov(pts.T))
        proj = pts @ vecs[:, -1]
        projw = pts @ vecs[:, 0]
        length_m = (proj.max() - proj.min() + 1.0) * pixel_m
        width_m = (projw.max() - projw.min() + 1.0) * pixel_m
        if length_m > 1500:    # troppo lungo anche per una fila di navi: struttura fissa
            continue
        objs.append((round(length_m), round(width_m)))
    return sorted(objs, reverse=True), sea_fraction


def classify(objs):
    """Da oggetti (lunghezza, larghezza) a stima del numero di navi.

    - oggetto largo >= 100 m e lungo >= 230 m: due navi affiancate in trasbordo (2 navi, 1 trasbordo)
    - oggetto lungo > 400 m: più navi in fila, contate lunghezza/300 m arrotondato (minimo 2)
    - altrimenti una nave
    Restituisce navi stimate >=180 m, >=250 m, trasbordi, oggetti fusi.
    """
    n180 = n250 = sts = merged = 0
    for length, width in objs:
        if length > MERGED_LEN:
            k = max(2, round(length / 300.0))
            merged += 1
            n180 += k
            n250 += k
        elif width >= STS_MIN_WIDTH and length >= STS_MIN_LEN:
            sts += 1
            n180 += 2
            n250 += 2 if length >= LEN_VLCC else 0
        else:
            n180 += 1 if length >= LEN_BIG else 0
            n250 += 1 if length >= LEN_VLCC else 0
    return n180, n250, sts, merged


# ---------------------------------------------------------------- archivio

def load_done():
    done = set()
    if CSV_PATH.exists():
        with CSV_PATH.open() as f:
            for row in csv.DictReader(f):
                done.add((row["aoi"], row["scene_id"]))
    return done


def migrate_csv():
    """Porta un CSV con le vecchie colonne al formato attuale (colonne nuove vuote)."""
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == CSV_FIELDS:
            return
        old = list(reader)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in old:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def append_rows(rows):
    DATA_DIR.mkdir(exist_ok=True)
    migrate_csv()
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def write_latest():
    if not CSV_PATH.exists():
        return
    latest = {}
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            prev = latest.get(row["aoi"])
            if prev is None or row["acq_utc"] > prev["acq_utc"]:
                latest[row["aoi"]] = row
    out = {
        "generato_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fonte": "Copernicus Sentinel-1 GRD via Copernicus Data Space Ecosystem, rilevamento CFAR",
        "nota": "Conteggio istantaneo di navi lunghe presenti nella zona al passaggio del satellite, incluse quelle con AIS spento. Indice di attività, non barili. Le righe precedenti al 29/09/2026 non hanno n_sts e n_merged e contano gli oggetti, non le navi stimate.",
        "legenda": "n_ge180m/n_ge250m = navi stimate (i trasbordi contano 2 navi); n_sts = coppie affiancate in trasbordo; n_merged = oggetti di più navi in fila. Zone: hormuz (stretto), yanbu e fujairah (terminal di bypass), sohar_sts (area trasbordi), ras_tanura, mina_ahmadi, basra, kharg, das (terminal di carico nel Golfo).",
        "zone": {k: {"acq_utc": v["acq_utc"], "n_ge180m": int(v["n_ge180m"] or 0),
                     "n_ge250m": int(v["n_ge250m"] or 0),
                     "n_sts": int(v["n_sts"]) if v.get("n_sts") not in (None, "") else None,
                     "n_merged": int(v["n_merged"]) if v.get("n_merged") not in (None, "") else None,
                     "sea_fraction": float(v["sea_fraction"])}
                 for k, v in latest.items()},
    }
    LATEST_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    import requests
    session = requests.Session()
    token = get_token(session)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=LOOKBACK_DAYS)
    done = load_done()
    rows = []
    for aoi, bbox in AOIS.items():
        try:
            scenes = search_scenes(session, token, bbox, start, now)
        except Exception as e:  # una zona che fallisce non blocca le altre
            print(f"[{aoi}] ricerca fallita: {e}")
            continue
        seen_times = set()
        for sc in sorted(scenes, key=lambda s: s["datetime"] or ""):
            acq = sc["datetime"]
            if not acq or (aoi, sc["id"]) in done:
                continue
            key = acq[:16]  # scene contigue della stessa orbita: una sola lettura
            if key in seen_times:
                continue
            seen_times.add(key)
            try:
                vv, valid = fetch_vv(session, token, bbox, acq)
            except Exception as e:
                print(f"[{aoi}] {acq} download fallito: {e}")
                continue
            if valid.mean() < 0.5:
                print(f"[{aoi}] {acq} copertura parziale ({valid.mean():.0%}), salto")
                continue
            objs, sea_frac = detect_ships(vv, valid)
            n180, n250, sts, merged = classify(objs)
            row = {
                "acq_utc": acq, "date_utc": acq[:10], "aoi": aoi, "scene_id": sc["id"],
                "n_ge180m": n180, "n_ge250m": n250, "n_sts": sts, "n_merged": merged,
                "n_objects": sum(1 for L, _ in objs if L >= LEN_BIG),
                "lengths_m": " ".join(f"{L}x{W}" for L, W in objs if L >= 120),
                "sea_fraction": f"{sea_frac:.2f}",
                "processed_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            print(f"[{aoi}] {acq}: {n180} navi >=180 m, {n250} >=250 m, {sts} trasbordi, {merged} oggetti fusi")
            rows.append(row)
    if rows:
        append_rows(rows)
    write_latest()
    print(f"Nuove righe: {len(rows)}")


if __name__ == "__main__":
    main()
