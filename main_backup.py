from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
import datetime as dt
import os
import re
import shutil
import sqlite3
import subprocess

import cv2
import numpy as np
import pytesseract


# ============================================================
# OSNOVNA PODEŠAVANJA
# ============================================================

app = FastAPI(title="Merna mesta - ESP32 OCR server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "pictures")
DB_PATH = os.path.join(BASE_DIR, "merenja.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/pictures", StaticFiles(directory=UPLOAD_FOLDER), name="pictures")


# Za svaki ESP možeš posebno podesiti crop i rotaciju.
# crop format: (x1, y1, x2, y2), vrednosti su procenti od slike.
# Ako je slika naopako, ostavi rotate: 180. Ako nije, stavi rotate: 0.
METER_CONFIG = {
    "esp_1": {
        "sat": "sat_1",
        "label": "Merno mesto 1",
        "crop": (0.28, 0.42, 0.78, 0.82),
        "digit_crop": (0.18, 0.22, 0.76, 0.82),
        "rotate": 0,
        "flip_horizontal": True,
        "min_brightness": 25,
        "min_contrast": 10,
        "decimal_digits": 2
    },
    "esp_2": {
        "sat": "sat_2",
        "label": "Merno mesto 2",
        "crop": (0.28, 0.42, 0.78, 0.82),
        "digit_crop": (0.18, 0.22, 0.76, 0.82),
        "rotate": 0,
        "flip_horizontal": True,
        "min_brightness": 25,
        "min_contrast": 10,
        "decimal_digits": 2
    }
}


# ============================================================
# BAZA
# ============================================================

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS merenja (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            esp_id TEXT,
            sat TEXT NOT NULL,
            vrednost REAL,
            vreme TEXT NOT NULL,
            original_slika TEXT,
            crop_slika TEXT,
            processed_slika TEXT,
            ocr_raw TEXT,
            ocr_ok INTEGER DEFAULT 0
        )
    """)
    conn.commit()

    # Migracija za slučaj da već postoji stara tabela bez svih kolona.
    postojece_kolone = {row["name"] for row in conn.execute("PRAGMA table_info(merenja)").fetchall()}
    potrebne_kolone = {
        "esp_id": "TEXT",
        "original_slika": "TEXT",
        "crop_slika": "TEXT",
        "processed_slika": "TEXT",
        "ocr_raw": "TEXT",
        "ocr_ok": "INTEGER DEFAULT 0"
    }

    for kolona, tip in potrebne_kolone.items():
        if kolona not in postojece_kolone:
            conn.execute(f"ALTER TABLE merenja ADD COLUMN {kolona} {tip}")

    conn.commit()
    conn.close()


init_db()


# ============================================================
# MODELI
# ============================================================

class PodatakSaSata(BaseModel):
    sat: str
    vrednost: float


# ============================================================
# POMOĆNE FUNKCIJE
# ============================================================

def detect_esp_id(filename: str) -> str:
    """Prepoznaje ESP na osnovu imena fajla: strujomer_esp_1.jpg ili strujomer_esp_2.jpg."""
    name = filename.lower()
    if "esp_2" in name or "sat_2" in name:
        return "esp_2"
    return "esp_1"


def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        name += ".jpg"
    return name


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def current_time() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def crop_and_preprocess(image_path: str, esp_id: str) -> Tuple[str, str]:
    """Pravi crop displeja i procesiranu crno-belu sliku za OCR."""
    cfg = METER_CONFIG[esp_id]

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("OpenCV ne može da učita sliku.")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = cfg["crop"]

    crop = img[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]

    if cfg["rotate"] == 180:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    elif cfg["rotate"] == 90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    elif cfg["rotate"] == -90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if cfg.get("flip_horizontal", False):
        crop = cv2.flip(crop, 1)

    dx1, dy1, dx2, dy2 = cfg.get("digit_crop", (0.0, 0.0, 1.0, 1.0))
    ch, cw = crop.shape[:2]
    crop = crop[int(ch * dy1):int(ch * dy2), int(cw * dx1):int(cw * dx2)]

    crop_name = f"latest_crop_{esp_id}.jpg"
    crop_path = os.path.join(UPLOAD_FOLDER, crop_name)
    cv2.imwrite(crop_path, crop)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Uvećanje pomaže OCR-u kod malih cifara.
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Lokalni kontrast.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blago čišćenje šuma.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarizacija.
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    # Tesseract bolje čita tamne cifre na beloj pozadini.
    # Ako je slika većinski tamna, okreni boje.
    if np.mean(processed) < 127:
        processed = cv2.bitwise_not(processed)

    # Malo zatvaranje prekida na segmentima cifara.
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    processed_name = f"latest_processed_{esp_id}.jpg"
    processed_path = os.path.join(UPLOAD_FOLDER, processed_name)
    cv2.imwrite(processed_path, processed)

    return crop_path, processed_path


def parse_value(raw_text: str, decimal_digits: int = 1) -> Optional[float]:
    """
    Izvlači vrednost iz OCR teksta.
    Ako OCR vrati 222, a decimal_digits=1, pretvara u 22.2.
    Ako vrati 55, pretvara u 5.5.
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    text = text.replace(",", ".")
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = text.replace("S", "5").replace("s", "5")

    candidates = re.findall(r"\d+\.\d+|\d+", text)
    if not candidates:
        return None

    token = candidates[0]

    try:
        if "." in token:
            return float(token)

        if decimal_digits > 0:
            if len(token) <= decimal_digits:
                return None
            token = token[:-decimal_digits] + "." + token[-decimal_digits:]

        return float(token)

    except ValueError:
        return None


def run_ssocr(image_path: str, decimal_digits: int) -> Tuple[Optional[float], str]:
    """Pokušava OCR preko SSOCR-a, ako je instaliran."""
    if shutil.which("ssocr") is None:
        return None, "SSOCR nije instaliran."

    commands = [
        ["ssocr", "-d", "-1", image_path],
        ["ssocr", "-d", "-1", "rle", "make_mono", image_path],
    ]

    raw_outputs = []

    for cmd in commands:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            out = (p.stdout or "").strip()
            err = (p.stderr or "").strip()
            raw_outputs.append(out if out else err)

            value = parse_value(out, decimal_digits)
            if value is not None:
                return value, out
        except Exception as e:
            raw_outputs.append(str(e))

    return None, " | ".join(raw_outputs)


def run_tesseract(image_path: str, decimal_digits: int) -> Tuple[Optional[float], str]:
    """Fallback OCR preko Tesseract-a."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Tesseract: slika nije učitana."

        configs = [
            "--psm 7 -c tessedit_char_whitelist=0123456789.,",
            "--psm 8 -c tessedit_char_whitelist=0123456789.,",
            "--psm 13 -c tessedit_char_whitelist=0123456789.,",
        ]

        raw_outputs = []

        for cfg in configs:
            text = pytesseract.image_to_string(img, config=cfg).strip()
            raw_outputs.append(text)
            value = parse_value(text, decimal_digits)
            if value is not None:
                return value, text

        return None, " | ".join(raw_outputs)
    except Exception as e:
        return None, f"Tesseract greška: {e}"


def ocr_reading(image_path: str, esp_id: str) -> Dict:
    """Kompletna OCR obrada: crop, preprocessing, SSOCR, Tesseract fallback."""
    cfg = METER_CONFIG[esp_id]
    crop_path, processed_path = crop_and_preprocess(image_path, esp_id)

    decimal_digits = cfg.get("decimal_digits", 1)

    value, raw = run_ssocr(crop_path, decimal_digits)
    engine = "ssocr"

    if value is None:
        value, raw = run_tesseract(processed_path, decimal_digits)
        engine = "tesseract"

    return {
        "value": value,
        "raw": raw,
        "engine": engine,
        "crop_slika": os.path.basename(crop_path),
        "processed_slika": os.path.basename(processed_path),
        "ok": value is not None
    }


def insert_measurement(
    esp_id: str,
    sat: str,
    value: Optional[float],
    original_slika: str,
    crop_slika: Optional[str],
    processed_slika: Optional[str],
    ocr_raw: str,
    ok: bool
):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO merenja
        (esp_id, sat, vrednost, vreme, original_slika, crop_slika, processed_slika, ocr_raw, ocr_ok)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            esp_id,
            sat,
            value,
            current_time(),
            original_slika,
            crop_slika,
            processed_slika,
            ocr_raw,
            1 if ok else 0
        )
    )
    conn.commit()
    conn.close()


# ============================================================
# HTML DASHBOARD
# ============================================================

HTML_STRANICA = """
<!DOCTYPE html>
<html lang="sr">
<head>
    <meta charset="UTF-8">
    <title>Merna mesta - Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #eef2f3;
            margin: 0;
            padding: 20px;
            color: #2c3e50;
            text-align: center;
        }
        h1 { margin-bottom: 10px; }
        .status { color: #555; margin-bottom: 20px; }
        .kontejner {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .kartica, .grafikon-kartica, .kamera-kartica {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.07);
            padding: 18px;
        }
        .kartica {
            width: 260px;
        }
        .vrednost {
            font-size: 38px;
            font-weight: 700;
            margin-top: 10px;
        }
        .kamere-kontejner {
            display: flex;
            justify-content: center;
            gap: 20px;
            max-width: 95%;
            margin: 25px auto;
            flex-wrap: wrap;
        }
        .kamera-kartica {
            flex: 1;
            min-width: 320px;
            max-width: 46%;
        }
        .kamera-kartica img {
            width: 100%;
            max-height: 280px;
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid #ddd;
            background: #fafafa;
        }
        .grafikon-kartica {
            width: 90%;
            margin: 30px auto;
        }
        .ocr {
            font-size: 13px;
            color: #666;
            margin-top: 8px;
            word-break: break-word;
        }
        button {
            margin-top: 20px;
            padding: 10px 25px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 15px;
        }
    </style>
</head>
<body>
    <h1>Merna mesta - Kontrolna tabla</h1>
    <div class="status" id="status">Čekam podatke...</div>

    <div class="kontejner">
        <div class="kartica">
            <h3>Merno mesto 1</h3>
            <div id="val-sat_1" class="vrednost" style="color:#2ecc71;">--</div>
        </div>
        <div class="kartica">
            <h3>Merno mesto 2</h3>
            <div id="val-sat_2" class="vrednost" style="color:#3498db;">--</div>
        </div>
    </div>

    <div class="kamere-kontejner">
        <div class="kamera-kartica">
            <h4>Merno mesto 1 - crop za OCR</h4>
            <img id="slikaKamere1" src="" alt="Čeka se slika sa ESP 1">
            <div class="ocr" id="ocr1"></div>
        </div>
        <div class="kamera-kartica">
            <h4>Merno mesto 2 - crop za OCR</h4>
            <img id="slikaKamere2" src="" alt="Čeka se slika sa ESP 2">
            <div class="ocr" id="ocr2"></div>
        </div>
    </div>

    <div class="grafikon-kartica">
        <canvas id="mojGrafikon" style="width:100%; max-height:420px;"></canvas>
        <button onclick="ocistiBazu()">🗑️ Obriši sve podatke</button>
    </div>

    <script>
        const ctx = document.getElementById('mojGrafikon').getContext('2d');

        const grafikon = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Merno mesto 1',
                        data: [],
                        borderColor: '#2ecc71',
                        backgroundColor: '#2ecc71',
                        tension: 0.2
                    },
                    {
                        label: 'Merno mesto 2',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: '#3498db',
                        tension: 0.2
                    }
                ]
            },
            options: {
                parsing: false,
                responsive: true,
                scales: {
                    x: {
                        type: 'category',
                        title: { display: true, text: 'Vreme' }
                    },
                    y: {
                        title: { display: true, text: 'Očitana vrednost' }
                    }
                }
            }
        });

        function formatVrednost(v) {
            if (v === null || v === undefined) return "--";
            return Number(v).toFixed(1);
        }

        function setImage(id, filename) {
            const img = document.getElementById(id);
            if (filename) {
                img.src = "/pictures/" + filename + "?t=" + new Date().getTime();
            }
        }

        async function osveziPodatke() {
            try {
                const odgovor = await fetch('/api/podaci');
                const podaci = await odgovor.json();

                document.getElementById('status').innerText =
                    "Poslednje osvežavanje: " + new Date().toLocaleTimeString();

                document.getElementById('val-sat_1').innerText = formatVrednost(podaci.trenutno.sat_1);
                document.getElementById('val-sat_2').innerText = formatVrednost(podaci.trenutno.sat_2);

                setImage("slikaKamere1", podaci.slike.esp_1.crop_slika);
                setImage("slikaKamere2", podaci.slike.esp_2.crop_slika);

                document.getElementById("ocr1").innerText =
                    podaci.slike.esp_1.ocr_raw ? "OCR: " + podaci.slike.esp_1.ocr_raw : "";
                document.getElementById("ocr2").innerText =
                    podaci.slike.esp_2.ocr_raw ? "OCR: " + podaci.slike.esp_2.ocr_raw : "";

                grafikon.data.datasets[0].data = podaci.istorija.sat_1.map(p => ({ x: p.vreme, y: p.vrednost }));
                grafikon.data.datasets[1].data = podaci.istorija.sat_2.map(p => ({ x: p.vreme, y: p.vrednost }));

                grafikon.update();
            } catch (e) {
                console.error(e);
                document.getElementById('status').innerText = "Greška pri osvežavanju podataka.";
            }
        }

        async function ocistiBazu() {
            if (!confirm("Sigurno želiš da obrišeš sva merenja?")) return;
            await fetch('/api/obrisi', { method: 'DELETE' });
            await osveziPodatke();
        }

        setInterval(osveziPodatke, 3000);
        osveziPodatke();
    </script>
</body>
</html>
"""


# ============================================================
# ENDPOINTI
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTML_STRANICA


@app.post("/upload")
async def primi_sliku(file: UploadFile = File(...)):
    original_filename = safe_filename(file.filename or "strujomer_nepoznato.jpg")
    esp_id = detect_esp_id(original_filename)

    if esp_id not in METER_CONFIG:
        raise HTTPException(status_code=400, detail="Nepoznat ESP ID.")

    cfg = METER_CONFIG[esp_id]
    sat = cfg["sat"]

    # Čuvamo i timestamp verziju i latest verziju.
    ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
    ts_name = f"{timestamp()}_{esp_id}{ext}"
    latest_name = f"latest_original_{esp_id}{ext}"

    ts_path = os.path.join(UPLOAD_FOLDER, ts_name)
    latest_path = os.path.join(UPLOAD_FOLDER, latest_name)

    content = await file.read()
    with open(ts_path, "wb") as f:
        f.write(content)
    with open(latest_path, "wb") as f:
        f.write(content)

    print(f"[{current_time()}] Primljena slika: {original_filename} -> {esp_id}")

    ocr_result = {
        "value": None,
        "raw": "",
        "engine": "none",
        "crop_slika": None,
        "processed_slika": None,
        "ok": False
    }

    try:
        ocr_result = ocr_reading(latest_path, esp_id)
        print(
            f"[OCR] esp={esp_id}, engine={ocr_result['engine']}, "
            f"value={ocr_result['value']}, raw={ocr_result['raw']}"
        )
    except Exception as e:
        ocr_result["raw"] = str(e)
        print(f"[OCR GREŠKA] {e}")

    insert_measurement(
        esp_id=esp_id,
        sat=sat,
        value=ocr_result["value"],
        original_slika=latest_name,
        crop_slika=ocr_result.get("crop_slika"),
        processed_slika=ocr_result.get("processed_slika"),
        ocr_raw=ocr_result.get("raw", ""),
        ok=ocr_result.get("ok", False)
    )

    return {
        "status": "uspeh" if ocr_result["ok"] else "primljeno_ali_ocr_nije_siguran",
        "esp_id": esp_id,
        "sat": sat,
        "ocitano": ocr_result["value"],
        "ocr_raw": ocr_result["raw"],
        "engine": ocr_result["engine"],
        "slika": latest_name,
        "crop": ocr_result.get("crop_slika"),
        "processed": ocr_result.get("processed_slika")
    }


@app.post("/update")
async def azuriraj_rucno(podatak: PodatakSaSata):
    if podatak.sat not in ["sat_1", "sat_2"]:
        raise HTTPException(status_code=400, detail="Koristi 'sat_1' ili 'sat_2'.")

    insert_measurement(
        esp_id="manual",
        sat=podatak.sat,
        value=podatak.vrednost,
        original_slika=None,
        crop_slika=None,
        processed_slika=None,
        ocr_raw="manual",
        ok=True
    )

    return {"status": "uspeh", "sat": podatak.sat, "vrednost": podatak.vrednost}


@app.get("/api/podaci")
async def api_podaci():
    conn = get_conn()

    rows = conn.execute("""
        SELECT sat, vrednost, vreme
        FROM merenja
        WHERE vrednost IS NOT NULL
        ORDER BY id DESC
        LIMIT 80
    """).fetchall()

    latest_rows = conn.execute("""
        SELECT *
        FROM merenja
        WHERE id IN (
            SELECT MAX(id)
            FROM merenja
            GROUP BY esp_id
        )
    """).fetchall()

    conn.close()

    istorija = {"sat_1": [], "sat_2": []}
    trenutno = {"sat_1": None, "sat_2": None}

    seen_current = set()
    for row in rows:
        if row["sat"] not in seen_current:
            trenutno[row["sat"]] = row["vrednost"]
            seen_current.add(row["sat"])

    for row in reversed(rows):
        if row["sat"] in istorija:
            istorija[row["sat"]].append({
                "vreme": row["vreme"],
                "vrednost": row["vrednost"]
            })

    slike = {
        "esp_1": {"original_slika": None, "crop_slika": None, "processed_slika": None, "ocr_raw": None},
        "esp_2": {"original_slika": None, "crop_slika": None, "processed_slika": None, "ocr_raw": None}
    }

    for row in latest_rows:
        esp_id = row["esp_id"]
        if esp_id in slike:
            slike[esp_id] = {
                "original_slika": row["original_slika"],
                "crop_slika": row["crop_slika"],
                "processed_slika": row["processed_slika"],
                "ocr_raw": row["ocr_raw"]
            }

    return {
        "trenutno": trenutno,
        "istorija": istorija,
        "slike": slike
    }


@app.delete("/api/obrisi")
async def obrisi_bazu():
    conn = get_conn()
    conn.execute("DELETE FROM merenja")
    conn.commit()
    conn.close()
    return {"status": "baza obrisana"}


@app.get("/api/status")
async def status():
    return {
        "server": "radi",
        "ssocr_installed": shutil.which("ssocr") is not None,
        "tesseract_installed": shutil.which("tesseract") is not None,
        "pictures_folder": UPLOAD_FOLDER
    }
# test
