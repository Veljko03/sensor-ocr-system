from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
import datetime as dt
import math
import os
import re
import shutil
import sqlite3
import subprocess
import asyncio
import time
import logging
from collections import Counter
from contextlib import AsyncExitStack

import cv2
import numpy as np
import pytesseract
from frame_confirmation import FrameConfirmation
from image_ingest import identify_device, read_bounded_upload, replace_latest
from upload_limits import UploadBodyLimit
import ocr_candidate


# ============================================================
# OSNOVNA PODEŠAVANJA
# ============================================================

app = FastAPI(title="Festo / Strujomer OCR server")

SERVER_VERSION = "festo-structural-v4"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.environ.get("SENSOR_OCR_DATA_DIR", BASE_DIR))
UPLOAD_FOLDER = os.path.join(DATA_DIR, "pictures")
DB_PATH = os.path.join(DATA_DIR, "merenja.db")
cv2.setNumThreads(1)
if ocr_candidate.MODE != "dynamic":
    raise RuntimeError("The main server requires the verified dynamic OCR mode")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/pictures", StaticFiles(directory=UPLOAD_FOLDER), name="pictures")


# ============================================================
# PODEŠAVANJA ZA ESP KAMERE
# ============================================================
# crop = prvi, širi crop iz cele slike
# digit_crop = drugi crop unutar tog crop-a, idealno samo cifre bez "bar", "Auto", ivica itd.
# decimal_digits=2 znači: ako OCR vrati 232 -> 2.32, 655 -> 6.55

METER_CONFIG = {
    "esp_1": {
        "sat": "sat_1",
        "label": "Pritisak",

        # Prvi crop preko CELE slike. Za početak ostavi 0..1.
        # Ako je kamera trajno montirana, production varijanta može biti uža.
        "crop": (0.0, 0.0, 1.0, 1.0),
        "rotate": 0,
        "flip_horizontal": True,

        # Crop UNUTAR perspektivno ispravljenog PLAVOG LCD-a.
        # Ovo je podešeno za Festo simulator/reference koju koristimo:
        # izbacuje OutA/OutD odozgo i "bar" odozdo.
"digit_crop": (0.00, 0.16, 1.00, 0.62),
        "decimal_digits": 2,
        "min_value": 0.0,
        "max_value": 9.99,

        # Nemoj previsoko za ESP32 kameru.
        "min_sharpness": 1.0,
        "min_contrast": 6.0,

        # Detekcija plavog LCD-a. Sniženo da radi i kada je ekran mali
        # u kadru ili je fotografija malo isprana.
        "blue_dominance_threshold": 22,
        "blue_min_value": 120,
        "blue_min_area_fraction": 0.0015,
        "blue_max_area_fraction": 0.65,
    },
    "esp_2": {
        "sat": "sat_2",
        "label": "Protok",
        "crop": (0.0, 0.0, 1.0, 1.0),
        "rotate": 0,
        "flip_horizontal": True,
"digit_crop": (0.00, 0.16, 1.00, 0.62),
        # VAŽNO ZA TRENUTNI TEST:
        # simulator sa slike prikazuje npr. 6.04, dakle DVE decimale.
        # Ako pravi merni uređaj na esp_2 stvarno ima jednu decimalu,
        # vrati ovo na 1 tek kada pređeš sa simulatora na pravi uređaj.
        "decimal_digits": 2,
        "min_value": 0.0,
        "max_value": 200.0,

        "min_sharpness": 1.0,
        "min_contrast": 6.0,
        "blue_dominance_threshold": 22,
        "blue_min_value": 120,
        "blue_min_area_fraction": 0.0015,
        "blue_max_area_fraction": 0.65,
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

    # Ako već postoji stara tabela, dodaj kolone koje fale.
    postojece = {r["name"] for r in conn.execute("PRAGMA table_info(merenja)").fetchall()}
    dodatne = {
        "esp_id": "TEXT",
        "original_slika": "TEXT",
        "crop_slika": "TEXT",
        "processed_slika": "TEXT",
        "ocr_raw": "TEXT",
        "ocr_ok": "INTEGER DEFAULT 0"
    }
    for kolona, tip in dodatne.items():
        if kolona not in postojece:
            conn.execute(f"ALTER TABLE merenja ADD COLUMN {kolona} {tip}")

    conn.commit()
    conn.close()


init_db()
print(f"[START] OCR server version: {SERVER_VERSION}")

# Vreme pokretanja servera — koristimo ga da kartice ne pokazuju stare vrednosti iz prethodnih sesija
SERVER_START_VREME = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    name = filename.lower()
    if "esp_2" in name or "sat_2" in name:
        return "esp_2"
    return "esp_1"


def safe_filename(name: str) -> str:
    name = os.path.basename(name or "strujomer_nepoznato.jpg")
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        name += ".jpg"
    return name


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def current_time() -> str:
    # Čuvamo pun datum+vreme da bismo mogli da filtriramo po sesiji i prikažemo datum na grafikonu
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def apply_ratio_crop(img, ratio_crop: Tuple[float, float, float, float]):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = ratio_crop
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return img[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]


def normalize_orientation(crop, cfg: Dict):
    if cfg.get("rotate") == 180:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    elif cfg.get("rotate") == 90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    elif cfg.get("rotate") == -90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if cfg.get("flip_horizontal", False):
        crop = cv2.flip(crop, 1)

    return crop


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def warp_ocr_screen(img: np.ndarray, pts: np.ndarray,
                    out_w: int = 360, out_h: int = 420) -> np.ndarray:
    """
    Perspektivno ispravlja plavi Festo ekran.
    Dimenzije su namerno standardizovane da digit_crop radi isto
    čak i kada je kamera malo ukoso.
    """
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (out_w, out_h))


def find_ocr_screen(img: np.ndarray, cfg: Dict, esp_id: str) -> Tuple[np.ndarray, Dict]:
    """
    Nalazi PLAVI Festo LCD u dva nivoa:

    1) STRONG maska:
       traži svetao, zasićen, jasno plav region. Ovo odlično odvaja LCD
       od sivog kućišta i plavih dugmadi čak i kad je uređaj mali u kadru.

    2) BROAD maska:
       ako je slika tamnija / isprana, koristi blažu blue-dominance + HSV
       detekciju i zatim geometrijski filtrira kandidate.

    Rezultat se perspective-warp-uje na 360x420, pa dalji OCR više ne zavisi
    mnogo od udaljenosti i malog ugla kamere.
    """
    if img is None or img.size == 0:
        raise ValueError("Prazna slika u find_ocr_screen().")

    h, w = img.shape[:2]
    full_area = float(h * w)

    b, g, r = cv2.split(img)
    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)
    blue_score = b16 - ((g16 + r16) // 2)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value_channel = cv2.split(hsv)

    dominance_thr = int(cfg.get("blue_dominance_threshold", 22))
    min_blue_value = int(cfg.get("blue_min_value", 120))
    min_frac = float(cfg.get("blue_min_area_fraction", 0.0015))
    max_frac = float(cfg.get("blue_max_area_fraction", 0.65))

    # --------------------------------------------------------
    # A) STRONG maska - prvo pokušaj najčistiju detekciju LCD-a
    # --------------------------------------------------------
    strong_dom = max(45, dominance_thr + 20)
    strong_value = max(175, min_blue_value + 45)

    strong_mask = np.where(
        (blue_score >= strong_dom)
        & (value_channel >= strong_value)
        & (saturation >= 42),
        255, 0
    ).astype(np.uint8)

    strong_mask = cv2.morphologyEx(
        strong_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    strong_mask = cv2.morphologyEx(
        strong_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    # --------------------------------------------------------
    # B) BROAD maska - fallback za tamniji / ispraniji kadar
    # --------------------------------------------------------
    mask_dom = np.where(
        (blue_score >= dominance_thr)
        & (value_channel >= min_blue_value),
        255, 0
    ).astype(np.uint8)

    mask_hsv = cv2.inRange(
        hsv,
        np.array([88, 45, min_blue_value], dtype=np.uint8),
        np.array([145, 255, 255], dtype=np.uint8),
    )

    broad_mask = cv2.bitwise_or(mask_dom, mask_hsv)
    broad_mask = cv2.morphologyEx(
        broad_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2
    )
    broad_mask = cv2.morphologyEx(
        broad_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    def collect_candidates(mask: np.ndarray, strong: bool) -> List[Dict]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        image_cx = w / 2.0
        image_cy = h / 2.0
        image_diag = max(1.0, math.hypot(w, h))

        found = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            frac = area / full_area

            local_min_frac = min_frac * (0.55 if strong else 1.0)

            if frac < local_min_frac or frac > max_frac:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), _ = rect

            if rw <= 3 or rh <= 3:
                continue

            short_side = min(rw, rh)
            long_side = max(rw, rh)
            aspect = short_side / max(long_side, 1.0)

            # LCD je blago portrait / skoro kvadrat.
            if not (0.42 <= aspect <= 1.0):
                continue

            rect_area = max(1.0, rw * rh)
            fill_ratio = area / rect_area

            min_fill = 0.48 if strong else 0.32
            if fill_ratio < min_fill:
                continue

            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(
                contour_mask, [contour], -1, 255, thickness=-1
            )

            mean_blue = float(
                cv2.mean(
                    blue_score.astype(np.float32),
                    mask=contour_mask
                )[0]
            )

            center_dist = (
                math.hypot(cx - image_cx, cy - image_cy) / image_diag
            )
            center_bonus = max(
                0.72,
                1.0 - 0.55 * center_dist
            )

            score = (
                area
                * (0.65 + 0.35 * fill_ratio)
                * center_bonus
                * (1.0 + max(0.0, mean_blue) / 255.0)
            )

            found.append({
                "score": score,
                "contour": contour,
                "area_fraction": frac,
                "fill_ratio": fill_ratio,
                "aspect": aspect,
                "mean_blue": mean_blue,
                "center": (float(cx), float(cy)),
            })

        return found

    strong_candidates = collect_candidates(strong_mask, strong=True)

    if strong_candidates:
        candidates = strong_candidates
        selected_mask = strong_mask
        detection_mode = "strong"
    else:
        candidates = collect_candidates(broad_mask, strong=False)
        selected_mask = broad_mask
        detection_mode = "broad"

    mask_name = f"latest_blue_mask_{esp_id}.png"
    mask_path = os.path.join(UPLOAD_FOLDER, mask_name)
    cv2.imwrite(mask_path, selected_mask)

    strong_name = f"latest_blue_mask_strong_{esp_id}.png"
    cv2.imwrite(
        os.path.join(UPLOAD_FOLDER, strong_name),
        strong_mask
    )

    broad_name = f"latest_blue_mask_broad_{esp_id}.png"
    cv2.imwrite(
        os.path.join(UPLOAD_FOLDER, broad_name),
        broad_mask
    )

    if not candidates:
        raise ValueError(
            "PLAVI LCD NIJE PRONAĐEN. "
            f"Proveri /pictures/{strong_name} i /pictures/{broad_name}"
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]

    # Strong maska uglavnom već sadrži samo LCD.
    # Kod broad maske može postojati uzak plavi 'rep' do dugmadi,
    # pa tada trimujemo na najduži kontinualni široki deo.
    if detection_mode == "strong":
        rect = cv2.minAreaRect(best["contour"])
    else:
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(
            contour_mask, [best["contour"]], -1, 255, thickness=-1
        )

        row_counts = np.count_nonzero(contour_mask, axis=1)
        max_row = int(row_counts.max()) if row_counts.size else 0
        refined_mask = contour_mask.copy()

        if max_row > 0:
            wide_flags = row_counts >= (0.70 * max_row)
            runs = []
            run_start = None

            for idx, is_wide in enumerate(wide_flags):
                if is_wide and run_start is None:
                    run_start = idx

                if run_start is not None and (
                    not is_wide or idx == len(wide_flags) - 1
                ):
                    run_end = idx if not is_wide else idx + 1
                    runs.append((run_start, run_end))
                    run_start = None

            if runs:
                y_lo, y_hi_exclusive = max(
                    runs, key=lambda rr: rr[1] - rr[0]
                )

                if (y_hi_exclusive - y_lo) >= 8:
                    refined_mask[:y_lo, :] = 0
                    refined_mask[y_hi_exclusive:, :] = 0

        refined_points = cv2.findNonZero(refined_mask)

        if refined_points is not None and len(refined_points) >= 20:
            rect = cv2.minAreaRect(refined_points)
        else:
            rect = cv2.minAreaRect(best["contour"])

    box = cv2.boxPoints(rect).astype("float32")
    screen = warp_ocr_screen(img, box)

    screen_name = f"latest_screen_{esp_id}.jpg"
    screen_path = os.path.join(UPLOAD_FOLDER, screen_name)
    cv2.imwrite(screen_path, screen)

    debug = {
        "mode": detection_mode,
        "mask": mask_name,
        "strong_mask": strong_name,
        "broad_mask": broad_name,
        "screen": screen_name,
        "area_fraction": round(float(best["area_fraction"]), 5),
        "fill_ratio": round(float(best["fill_ratio"]), 3),
        "aspect": round(float(best["aspect"]), 3),
        "mean_blue": round(float(best["mean_blue"]), 1),
        "candidate_count": len(candidates),
    }

    return screen, debug

def _remove_tiny_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Uklanja sitne artefakte / tekst koji upadne u digit crop.
    Decimalnu tačku nije problem ukloniti jer decimalno mesto već znamo
    iz METER_CONFIG["decimal_digits"].
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def preprocess_digit_screen(screen: np.ndarray, esp_id: str) -> Dict:
    """
    Obrada za BELA 7-segment svetla na PLAVOM Festo LCD-u.

    Bitna promena u v3:
      - crop je uži i sadrži samo velike cifre
      - SSOCR dobija CRNE cifre na BELOJ pozadini, što je kompatibilnije
        sa starijim SSOCR buildovima
      - pravimo 3 threshold varijante zbog fotografisanja monitora
    """
    cfg = METER_CONFIG[esp_id]
    paths: Dict = {}

    digit_crop = apply_ratio_crop(
        screen,
        cfg.get("digit_crop", (0.03, 0.29, 0.98, 0.615))
    )

    if digit_crop.size == 0:
        raise ValueError("digit_crop je prazan. Proveri METER_CONFIG['digit_crop'].")

    crop_name = f"latest_crop_{esp_id}.jpg"
    crop_path = os.path.join(UPLOAD_FOLDER, crop_name)
    cv2.imwrite(crop_path, digit_crop)
    paths["crop"] = crop_path

    h, w = digit_crop.shape[:2]

    # Ako je ekran mali u kadru, povećaj malo više.
    if h < 90:
        scale = 5.0
    elif h < 150:
        scale = 4.0
    elif h < 240:
        scale = 3.0
    else:
        scale = 2.0

    enlarged = cv2.resize(
        digit_crop, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Bela cifra: sva tri kanala su relativno visoka.
    # Plava pozadina: B je visok, ali R je znatno niži.
    white_score = np.min(enlarged, axis=2).astype(np.uint8)
    white_score = cv2.GaussianBlur(white_score, (3, 3), 0)

    otsu_threshold, _ = cv2.threshold(
        white_score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    base_threshold = int(np.clip(otsu_threshold, 90, 210))
    thresholds = [
        max(75, base_threshold - 18),
        base_threshold,
        min(225, base_threshold + 18),
    ]

    ssocr_paths = []
    white_on_black_paths = []

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    for idx, thr in enumerate(thresholds):
        white_on_black = np.where(white_score >= thr, 255, 0).astype(np.uint8)

        # Popuni samo mikro-prekide. Ne radimo agresivnu morfologiju
        # da ne spojimo susedne 7-seg cifre.
        white_on_black = cv2.morphologyEx(
            white_on_black, cv2.MORPH_CLOSE, close_kernel, iterations=1
        )

        min_area = max(10, int(white_on_black.size * 0.000025))
        white_on_black = _remove_tiny_components(white_on_black, min_area)

        wob_name = f"latest_segments_{esp_id}_{idx}.png"
        wob_path = os.path.join(UPLOAD_FOLDER, wob_name)
        cv2.imwrite(wob_path, white_on_black)
        white_on_black_paths.append(wob_path)

        # SSOCR/Tesseract u ovoj konfiguraciji dobijaju crne cifre na belom.
        black_on_white = cv2.bitwise_not(white_on_black)

        variant_name = f"latest_ssocr_{esp_id}_{idx}.png"
        variant_path = os.path.join(UPLOAD_FOLDER, variant_name)
        cv2.imwrite(variant_path, black_on_white)
        ssocr_paths.append(variant_path)

    paths["ssocr_variants"] = ssocr_paths
    paths["segment_variants"] = white_on_black_paths

    # Srednja varijanta je dashboard processed.
    processed = cv2.imread(ssocr_paths[1], cv2.IMREAD_GRAYSCALE)
    processed_name = f"latest_processed_{esp_id}.png"
    processed_path = os.path.join(UPLOAD_FOLDER, processed_name)
    cv2.imwrite(processed_path, processed)

    paths["processed_dashboard"] = processed_path
    paths["processed"] = processed_path

    return paths


def crop_and_preprocess(image_path: str, esp_id: str) -> Dict:
    cfg = METER_CONFIG[esp_id]

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("OpenCV ne može da učita sliku.")

    img = apply_ratio_crop(img, cfg.get("crop", (0.0, 0.0, 1.0, 1.0)))
    img = normalize_orientation(img, cfg)

    normalized_name = f"latest_normalized_{esp_id}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, normalized_name), img)

    screen, detection_debug = find_ocr_screen(img, cfg, esp_id)
    paths = preprocess_digit_screen(screen, esp_id)

    paths["detection_debug"] = detection_debug
    paths["normalized"] = os.path.join(UPLOAD_FOLDER, normalized_name)
    paths["screen"] = os.path.join(UPLOAD_FOLDER, detection_debug["screen"])
    paths["mask"] = os.path.join(UPLOAD_FOLDER, detection_debug["mask"])
    return paths


def image_quality_metrics(image_path: str) -> Tuple[float, float]:
    """
    Vraća (sharpness, contrast).
    Sharpness = variance Laplacian-a; korisnije je od same brightness
    za fotografisani svetleći LCD.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return 0.0, 0.0

    sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
    contrast = float(np.std(img))
    return sharpness, contrast


# ============================================================
# DIREKTNI 7-SEGMENT DECODER (bez OCR font modela)
# ============================================================

SEVEN_SEGMENT_PATTERNS = {
    "0": (1, 1, 1, 1, 1, 1, 0),
    "1": (0, 1, 1, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1),
    "3": (1, 1, 1, 1, 0, 0, 1),
    "4": (0, 1, 1, 0, 0, 1, 1),
    "5": (1, 0, 1, 1, 0, 1, 1),
    "6": (1, 0, 1, 1, 1, 1, 1),
    "7": (1, 1, 1, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
    " ": (0, 0, 0, 0, 0, 0, 0),
}


def _segment_occupancy(mask: np.ndarray, x: float, y: float,
                       cell_w: float, cell_h: float) -> List[float]:
    """
    Uzorkuje 7 malih zona unutar jedne očekivane cifre:
        a
      f   b
        g
      e   c
        d

    ROI-jevi su namerno uski i nalaze se u sredini segmenata, pa su
    manje osetljivi na blur/glow nego contour-based OCR.
    """
    rois = [
        (0.25, 0.02, 0.75, 0.10),  # a
        (0.84, 0.15, 0.98, 0.42),  # b
        (0.84, 0.58, 0.98, 0.85),  # c
        (0.25, 0.90, 0.75, 0.98),  # d
        (0.02, 0.58, 0.16, 0.85),  # e
        (0.02, 0.15, 0.16, 0.42),  # f
        (0.25, 0.47, 0.75, 0.55),  # g
    ]

    h, w = mask.shape[:2]
    result = []

    for rx1, ry1, rx2, ry2 in rois:
        x1 = max(0, min(w, int(round(x + rx1 * cell_w))))
        x2 = max(0, min(w, int(round(x + rx2 * cell_w))))
        y1 = max(0, min(h, int(round(y + ry1 * cell_h))))
        y2 = max(0, min(h, int(round(y + ry2 * cell_h))))

        roi = mask[y1:y2, x1:x2]

        if roi.size == 0:
            result.append(0.0)
        else:
            result.append(float(np.mean(roi > 0)))

    return result


def _best_segment_pattern(occupancy: List[float]) -> Tuple[str, float]:
    """
    Vraća (znak, cost). Manji cost = bolji match.
    """
    best_digit = " "
    best_cost = float("inf")

    for digit, expected in SEVEN_SEGMENT_PATTERNS.items():
        cost = sum(
            (float(observed) - float(target)) ** 2
            for observed, target in zip(occupancy, expected)
        ) / 7.0

        if cost < best_cost:
            best_cost = cost
            best_digit = digit

    return best_digit, best_cost


def _decode_direct_7segment_image(screen: np.ndarray, cfg: Dict,
                                      orientation: str) -> Dict:
    """
    Dekoduje jednu orijentaciju screen slike i vraća detaljan rezultat.
    """
    white_score = np.min(screen, axis=2).astype(np.uint8)
    white_score = cv2.GaussianBlur(white_score, (3, 3), 0)

    otsu, _ = cv2.threshold(
        white_score, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    threshold = int(np.clip(float(otsu) + 15.0, 120, 205))
    mask = np.where(white_score >= threshold, 255, 0).astype(np.uint8)

    h, w = mask.shape[:2]

    cell_w = 0.190 * w
    cell_h = 0.390 * h
    step_x = 0.216 * w
    rightmost_x = 0.696 * w
    base_y = 0.180 * h

    slots = []

    for slot_index in range(4):
        base_x = rightmost_x - (3 - slot_index) * step_x
        best_for_slot = None

        for dx in (-12, -8, -4, 0, 4, 8, 12):
            for dy in (-10, -5, 0, 5, 10):
                occupancy = _segment_occupancy(
                    mask,
                    base_x + dx,
                    base_y + dy,
                    cell_w,
                    cell_h
                )

                digit, cost = _best_segment_pattern(occupancy)

                candidate = {
                    "digit": digit,
                    "cost": float(cost),
                    "occupancy": [round(v, 3) for v in occupancy],
                    "dx": dx,
                    "dy": dy,
                }

                if best_for_slot is None or candidate["cost"] < best_for_slot["cost"]:
                    best_for_slot = candidate

        slots.append(best_for_slot)

    chars = [slot["digit"] for slot in slots]
    costs = [float(slot["cost"]) for slot in slots]

    # Leading blanks su normalni kada broj ima manje od 4 cifre.
    while chars and chars[0] == " ":
        chars.pop(0)
        costs.pop(0)

    result = {
        "orientation": orientation,
        "value": None,
        "digits": "",
        "avg_cost": 1.0,
        "max_cost": 1.0,
        "otsu": float(otsu),
        "threshold": threshold,
        "slots": slots,
        "reason": "",
    }

    if not chars:
        result["reason"] = "nema cifara"
        return result

    if any(ch == " " for ch in chars):
        result["reason"] = "unutrašnji blank slot"
        return result

    digits = "".join(chars)
    result["digits"] = digits

    decimal_digits = int(cfg.get("decimal_digits", 2))

    if len(digits) < decimal_digits + 1:
        result["reason"] = (
            f"premalo cifara za decimal_digits={decimal_digits}"
        )
        return result

    avg_cost = float(np.mean(costs)) if costs else 1.0
    max_cost = float(max(costs)) if costs else 1.0
    result["avg_cost"] = avg_cost
    result["max_cost"] = max_cost

    if avg_cost > 0.20 or max_cost > 0.32:
        result["reason"] = "slab geometrijski match"
        return result

    try:
        value = int(digits) / (10 ** decimal_digits)
    except ValueError:
        result["reason"] = "digits se ne mogu parsirati"
        return result

    min_value = cfg.get("min_value")
    max_value = cfg.get("max_value")

    if min_value is not None and value < float(min_value):
        result["reason"] = f"value ispod min={min_value}"
        return result

    if max_value is not None and value > float(max_value):
        result["reason"] = f"value iznad max={max_value}"
        return result

    result["value"] = float(value)
    result["reason"] = "ok"
    return result


def run_direct_7segment(screen_path: str, cfg: Dict) -> Tuple[Optional[float], str]:
    """
    Direktni OpenCV 7-segment dekoder.

    Automatski proba:
      - screen kako je došao iz pipeline-a
      - horizontalno ogledalo

    Ovo znači da čak i ako je flip_horizontal slučajno pogrešno podešen,
    direct7seg često može sam da izabere ispravnu orijentaciju.
    """
    screen = cv2.imread(screen_path)

    if screen is None or screen.size == 0:
        return None, "direct7seg: screen nije učitan."

    variants = [
        ("normal", screen),
        ("mirror", cv2.flip(screen, 1)),
    ]

    results = [
        _decode_direct_7segment_image(img_variant, cfg, orientation)
        for orientation, img_variant in variants
    ]

    valid = [r for r in results if r["value"] is not None]

    if valid:
        best = min(valid, key=lambda r: (r["avg_cost"], r["max_cost"]))
        confidence = max(0.0, min(1.0, 1.0 - best["avg_cost"]))

        raw = (
            f"direct7seg orientation={best['orientation']}, "
            f"digits={best['digits']}, value={best['value']}, "
            f"confidence={confidence:.3f}, "
            f"avg_cost={best['avg_cost']:.3f}, "
            f"max_cost={best['max_cost']:.3f}, "
            f"otsu={best['otsu']:.1f}, thr={best['threshold']}, "
            f"slots={best['slots']}"
        )
        return best["value"], raw

    compact = [
        {
            "orientation": r["orientation"],
            "digits": r["digits"],
            "reason": r["reason"],
            "avg_cost": round(r["avg_cost"], 3),
            "max_cost": round(r["max_cost"], 3),
            "otsu": round(r["otsu"], 1),
            "thr": r["threshold"],
        }
        for r in results
    ]

    return None, f"direct7seg neuspeh: {compact}"

def parse_value(raw_text: str, decimal_digits: int = 2,
                min_value: Optional[float] = None,
                max_value: Optional[float] = None) -> Optional[float]:
    """
    Strogo parsiranje za vrednosti sa 2 decimale.
    Prihvata: 232 -> 2.32, 655 -> 6.55, 12.34 -> 12.34
    Odbija: 8, 88, 83, 1.1, 9.8, 1-81.a1
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    text = text.replace(",", ".")
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = text.replace("S", "5").replace("s", "5")

    # Ako OCR ubaci slova, najčešće je to šum.
    if re.search(r"[A-Za-z]", text):
        return None

    candidates = re.findall(r"\d+\.\d+|\d+", text)
    if not candidates:
        return None

    for token in candidates:
        try:
            if "." in token:
                whole, frac = token.split(".", 1)
                if len(frac) != decimal_digits:
                    continue
                if len(whole) < 1 or len(whole) > 2:
                    continue
                value = float(token)
            else:
                if decimal_digits > 0:
                    if len(token) not in (decimal_digits + 1, decimal_digits + 2):
                        continue
                    token = token[:-decimal_digits] + "." + token[-decimal_digits:]
                value = float(token)

            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue

            return value
        except ValueError:
            continue

    return None


def run_cmd(cmd: List[str]) -> Tuple[str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:
        return "", str(e)


def run_ssocr(image_path: str, cfg: Dict) -> Tuple[Optional[float], str]:
    """
    Primarni engine za seven-segment.
    Input je crna cifra na beloj pozadini.

    Koristimo samo opcije koje su bile prisutne i u tvom ranijem kodu,
    da ne zavisimo od novijih SSOCR buildova.
    """
    if shutil.which("ssocr") is None:
        return None, "SSOCR nije instaliran."

    decimal_digits = cfg.get("decimal_digits", 2)

    commands = [
        ["ssocr", "-d", "-1", image_path],
        ["ssocr", "-d", "-1", "-a", "-t", "25", image_path],
        ["ssocr", "-d", "-1", "-a", "-t", "40", image_path],
        ["ssocr", "-d", "-1", "-a", "-t", "55", image_path],
        ["ssocr", "-d", "-1", "-a", "-t", "70", image_path],
    ]

    raw_outputs = []

    for cmd in commands:
        out, err = run_cmd(cmd)
        raw_outputs.append(f"{' '.join(cmd)} => OUT:[{out}] ERR:[{err}]")

        value = parse_value(
            out,
            decimal_digits,
            cfg.get("min_value"),
            cfg.get("max_value")
        )
        if value is not None:
            return value, out

    return None, " || ".join(raw_outputs)


def run_tesseract(image_path: str, cfg: Dict) -> Tuple[Optional[float], str]:
    decimal_digits = cfg.get("decimal_digits", 2)

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

        for tc in configs:
            text = pytesseract.image_to_string(img, config=tc).strip()
            raw_outputs.append(f"{tc} => [{text}]")
            value = parse_value(text, decimal_digits, cfg.get("min_value"), cfg.get("max_value"))
            if value is not None:
                return value, text

        return None, " || ".join(raw_outputs)
    except Exception as e:
        return None, f"Tesseract greška: {e}"


def ocr_reading(image_path: str, esp_id: str) -> Dict:
    """
    Produkcioni OCR za realni Festo merac.

    esp_1:
      - detekcija plavog LCD-a
      - perspective correction
      - 3 fiksna digit slota
      - calibrated 7-segment decoder
      - confidence rejection

    Stari SSOCR/Tesseract pipeline se ne koristi.
    """

    if esp_id != "esp_1":
        return {
            "value": None,
            "raw": f"ODBIJENO: novi real-meter OCR nije konfigurisan za {esp_id}",
            "engine": "none",
            "crop_slika": None,
            "processed_slika": None,
            "ok": False
        }

    # Decode immutable ingress bytes; old helper callers may still use a path.
    image = (cv2.imdecode(np.frombuffer(image_path, np.uint8), cv2.IMREAD_COLOR)
             if isinstance(image_path, bytes) else cv2.imread(image_path))

    if image is None:
        return {
            "value": None,
            "raw": "ODBIJENO: originalna slika nije mogla da se ucita.",
            "engine": "none",
            "crop_slika": None,
            "processed_slika": None,
            "ok": False
        }

    value, debug = ocr_candidate.read_real_meter_image(
        image,
        confidence_threshold=0.45
    )

    if value is None:
        return {
            "value": None,
            "raw": str(debug)[:8192],
            "engine": "structural_7seg",
            "crop_slika": None,
            "processed_slika": None,
            "ok": False
        }

    return {
        "value": value,
        "raw": str(debug)[:8192],
        "engine": "structural_7seg",
        "crop_slika": None,
        "processed_slika": None,
        "ok": True
    }


def insert_measurement(esp_id: str, sat: str, value: Optional[float],
                       original_slika: Optional[str], crop_slika: Optional[str],
                       processed_slika: Optional[str], ocr_raw: str, ok: bool):
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO merenja
            (esp_id, sat, vrednost, vreme, original_slika, crop_slika, processed_slika, ocr_raw, ocr_ok)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (esp_id, sat, value, current_time(), original_slika, crop_slika,
             processed_slika, ocr_raw, 1 if ok else 0),
        )
        conn.commit()
    finally:
        conn.close()


# ============================================================
# DASHBOARD
# ============================================================

HTML_STRANICA = """
<!DOCTYPE html>
<html lang="sr">
<head>
    <meta charset="UTF-8">
    <title>Merna mesta - OCR Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background:#eef2f3; margin:0; padding:20px; text-align:center; color:#2c3e50; }
        .cards, .images { display:flex; gap:20px; justify-content:center; flex-wrap:wrap; }
        .card { background:white; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,.08); padding:18px; width:330px; margin:10px; }
        .value { font-size:38px; font-weight:700; margin-top:12px; }
        .time { font-size:13px; color:#7f8c8d; margin-top:6px; }
        img { max-width:100%; border:1px solid #ddd; border-radius:8px; background:#fafafa; }
        .chart { background:white; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,.08); padding:18px; margin:25px auto; width:90%; }
        button { padding:10px 20px; border:0; border-radius:8px; color:white; background:#e74c3c; cursor:pointer; }
        .ocr { font-size:12px; color:#666; word-break:break-word; text-align:left; margin-top:8px; max-height:90px; overflow:auto; }
    </style>
</head>
<body>
    <h1>Merna mesta - OCR Dashboard</h1>
    <div id="status">Čekam podatke...</div>

    <div class="cards">
        <div class="card">
            <h3>Merno mesto 1</h3>
            <div id="val-sat_1" class="value" style="color:#2ecc71;">--</div>
            <div id="time-sat_1" class="time">Nema merenja</div>
        </div>
        <div class="card">
            <h3>Merno mesto 2</h3>
            <div id="val-sat_2" class="value" style="color:#3498db;">--</div>
            <div id="time-sat_2" class="time">Nema merenja</div>
        </div>
    </div>

    <div class="images">
        <div class="card"><h3>ESP 1 - originalna slika</h3><img id="img1" alt="Poslednja originalna fotografija sa ESP 1"><div id="raw1" class="ocr"></div></div>
        <div class="card"><h3>ESP 2 - originalna slika</h3><img id="img2" alt="Poslednja originalna fotografija sa ESP 2"><div id="raw2" class="ocr"></div></div>
    </div>

    <div class="chart">
        <canvas id="chart"></canvas>
        <br><button onclick="ocisti()">Obriši bazu</button>
    </div>

<script>
const ctx = document.getElementById('chart').getContext('2d');
const grafikon = typeof Chart === 'undefined' ? null : new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
    {
        label: 'Merno mesto 1',
        data: [],
        borderColor: '#2ecc71',
        backgroundColor: 'rgba(46,204,113,0.15)',
        tension: .2,
        spanGaps: true,
        pointRadius: 3,
        yAxisID: 'y'
    },
    {
        label: 'Merno mesto 2',
        data: [],
        borderColor: '#3498db',
        backgroundColor: 'rgba(52,152,219,0.15)',
        tension: .2,
        spanGaps: true,
        pointRadius: 3,
        yAxisID: 'y'
    }
]
    },
    options: {
        responsive: true,
       scales: {
    x: {
        title: { display: true, text: 'Datum / vreme' },
        ticks: { maxTicksLimit: 12, maxRotation: 30, autoSkip: true }
    },
    y: {
        type: 'linear',
        position: 'left',
        title: { display: true, text: 'Vrednost' },
        beginAtZero: false
    }
},
        plugins: {
            legend: { display: true },
            tooltip: { mode: 'index', intersect: false }
        }
    }
});

function fmt(v){ return (v === null || v === undefined) ? "-" : Number(v).toFixed(2); }
function img(id, name) {
    const el = document.getElementById(id);

    if (!name) {
        el.removeAttribute("src");
        return;
    }

    const url = "/pictures/" + name + "?v=" + Date.now();

    const preloaded = new Image();
    preloaded.onload = function () {
        el.src = url;
    };
    preloaded.src = url;
}

// Konvertuje "2024-06-22 10:30:00" → "22.06 10:30" za prikaz na x-osi
function fmtLabel(v) {
    if (!v) return '';
    const parts = v.split(' ');
    if (parts.length === 2) {
        const dp = parts[0].split('-');
        return dp[2] + '.' + dp[1] + ' ' + parts[1].substring(0, 5);
    }
    return v;
}

const states = {
    waiting: 'Čekam sliku', waiting_new_frame: 'Čekam novu sliku',
    processing: 'Čitam novu sliku', confirmed: 'Očitavanje potvrđeno',
    awaiting_confirmation: 'Čekam sledeću sliku za potvrdu',
    duplicate_frame: 'Stigla je ista slika; čekam novu',
    ocr_rejected: 'Displej nije pouzdano očitan', unsupported_device: 'OCR još nije podešen za ovaj uređaj',
    invalid_image: 'Primljen je neispravan JPEG', invalid_request: 'Upload je odbijen', image_storage_failed: 'Slika nije sačuvana',
    database_error: 'Merenje nije sačuvano', frames_too_close: 'Slike su stigle prebrzo',
};
async function showFrame(device, sat, imageId, rawId, frame, value) {
    const number = document.getElementById('val-' + sat);
    const photo = document.getElementById(imageId);
    number.innerText = '-';
    document.getElementById(rawId).innerText = states[frame.status] || 'Čekam novu sliku';
    document.getElementById('time-' + sat).innerText = frame.vreme ? 'Poslednja slika: ' + frame.vreme : 'Nema slike';
    if (!frame.sequence || !frame.original_slika) {
        photo.removeAttribute('src');
        if (photo.dataset.url) URL.revokeObjectURL(photo.dataset.url);
        delete photo.dataset.url;
        return;
    }
    const response = await fetch('/api/frame/' + device + '?sequence=' + frame.sequence, {cache: 'no-store'});
    if (response.status === 409) return;
    if (!response.ok) throw Error('Slika nije dostupna');
    const url = URL.createObjectURL(await response.blob());
    const decoded = new Image();
    try {
        decoded.src = url;
        await decoded.decode();
        const previous = photo.dataset.url;
        photo.src = url; photo.dataset.url = url;
        if (previous) URL.revokeObjectURL(previous);
        number.innerText = fmt(value);
    } catch (error) { URL.revokeObjectURL(url); throw error; }
}
let refreshing = false;
async function refresh(){
    if (refreshing) return;
    refreshing = true;
    try {
        const r = await fetch('/api/podaci?t=' + Date.now(), {cache: 'no-store'});
        if (!r.ok) throw Error('Server nije dostupan');
        const p = await r.json();
        await Promise.all([
            showFrame('esp_1', 'sat_1', 'img1', 'raw1', p.slike.esp_1, p.trenutno.sat_1),
            showFrame('esp_2', 'sat_2', 'img2', 'raw2', p.slike.esp_2, p.trenutno.sat_2),
        ]);
        document.getElementById('status').innerText = 'Poslednje osvežavanje: ' + new Date().toLocaleTimeString();
        if (grafikon) {
            const labels = Array.from(new Set([...p.istorija.sat_1.map(x => x.vreme), ...p.istorija.sat_2.map(x => x.vreme)])).sort();
            function valueAt(lista, vreme) {
                const item = lista.find(x => x.vreme === vreme);
                return item ? item.vrednost : null;
            }
            grafikon.data.labels = labels.map(fmtLabel);
            grafikon.data.datasets[0].data = labels.map(v => valueAt(p.istorija.sat_1, v));
            grafikon.data.datasets[1].data = labels.map(v => valueAt(p.istorija.sat_2, v));
            grafikon.update();
        }
    } catch(error) {
        document.getElementById('val-sat_1').innerText = '-';
        document.getElementById('val-sat_2').innerText = '-';
        document.getElementById('status').innerText = 'Veza sa serverom je prekinuta. Proveri terminal na RPi.';
    } finally { refreshing = false; }
}
async function ocisti(){ if(confirm("Obrisati sva merenja?")) { await fetch('/api/obrisi', {method:'DELETE'}); refresh(); } }
setInterval(refresh, 3000); refresh();
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


# ============================================================
# OCR FRAME CONFIRMATION
# ============================================================

OCR_CONFIRMATION = FrameConfirmation()
UPLOAD_LOCKS = {device: asyncio.Lock() for device in METER_CONFIG}
UPLOAD_SEQUENCE = {device: 0 for device in METER_CONFIG}
LIVE_FRAMES = {}
logger = logging.getLogger(__name__)


def invalidate_current(device, sequence, reason):
    OCR_CONFIRMATION.invalidate(device)
    previous = LIVE_FRAMES.get(device, {})
    LIVE_FRAMES[device] = dict(previous, sequence=sequence, value=None,
                               ocr_ok=False, status=reason)


async def reject_upload_body(scope):
    # A rejected multipart body may not identify a trustworthy camera.
    # Reset active confirmations rather than retaining a valid-looking number.
    for device in METER_CONFIG:
        async with UPLOAD_LOCKS[device]:
            OCR_CONFIRMATION.invalidate(device)
            if device in LIVE_FRAMES:
                UPLOAD_SEQUENCE[device] += 1
                invalidate_current(device, UPLOAD_SEQUENCE[device], "invalid_request")


app.add_middleware(UploadBodyLimit, on_rejected=reject_upload_body)


@app.post("/upload")
async def primi_sliku(file: UploadFile = File(...), esp: Optional[str] = None):
    received_at = time.monotonic()
    try:
        esp_id = identify_device(file.filename, esp)
    except ValueError:
        await file.close()
        raise HTTPException(400, "Fajl ili esp parametar mora označiti esp_1 ili esp_2.")
    UPLOAD_SEQUENCE[esp_id] += 1
    sequence = UPLOAD_SEQUENCE[esp_id]
    sat = METER_CONFIG[esp_id]["sat"]
    try:
        content, fingerprint = await read_bounded_upload(file)
    except (ValueError, OSError):
        async with UPLOAD_LOCKS[esp_id]:
            if sequence > LIVE_FRAMES.get(esp_id, {}).get("sequence", 0):
                invalidate_current(esp_id, sequence, "invalid_image")
                await asyncio.to_thread(insert_measurement, esp_id, sat, None,
                                        None, None, None, "invalid_image", False)
        raise HTTPException(400, "Primljen je nevažeći, nepotpun ili preveliki JPEG.")

    async with UPLOAD_LOCKS[esp_id]:
        if sequence <= LIVE_FRAMES.get(esp_id, {}).get("sequence", 0):
            raise HTTPException(409, "Starija slika ne može zameniti noviju.")
        try:
            latest = await asyncio.to_thread(replace_latest, UPLOAD_FOLDER, esp_id, content)
        except (OSError, ValueError):
            invalidate_current(esp_id, sequence, "image_storage_failed")
            raise HTTPException(503, "Slika nije mogla bezbedno da se sačuva.")
        LIVE_FRAMES[esp_id] = {
            "sequence": sequence, "value": None, "ocr_ok": False,
            "status": "processing", "original_slika": latest.name,
            "image": content, "vreme": current_time(),
            "received_at": received_at, "latency_ms": None,
        }
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(ocr_reading, content, esp_id)
        except Exception:
            logger.exception("OCR processing failed for %s", esp_id)
            result = {"value": None, "raw": "ocr_error", "engine": "structural_7seg", "ok": False}
        decision = OCR_CONFIRMATION.observe(
            esp_id, result.get("value") if result.get("ok") else None,
            fingerprint, sequence, received_at, time.monotonic(),
        )
        raw = (result.get("raw", "") + " | " + decision.reason)[:8192]
        try:
            await asyncio.to_thread(
                insert_measurement, esp_id, sat, decision.value, latest.name,
                None, None, raw, decision.confirmed,
            )
        except sqlite3.Error:
            invalidate_current(esp_id, sequence, "database_error")
            logger.exception("Measurement storage failed for %s", esp_id)
            raise HTTPException(503, "Merenje nije moglo da se sačuva.")
        LIVE_FRAMES[esp_id].update(
            value=decision.value, ocr_ok=decision.confirmed, status=decision.reason,
            ocr_raw=raw, latency_ms=round((time.monotonic()-started)*1000, 1),
        )
        return {
            "status": "uspeh" if decision.confirmed else "primljeno_ali_ocr_nije_siguran",
            "esp_id": esp_id, "sat": sat, "ocitano": decision.value,
            "ocr_raw": raw, "engine": result.get("engine", "none"),
            "slika": latest.name, "crop": None, "processed": None,
            "confirmation": decision.reason,
        }


@app.get("/api/frame/{esp_id}")
async def original_frame(esp_id: str, sequence: int):
    if esp_id not in METER_CONFIG:
        raise HTTPException(404, "Nepoznat ESP ID.")
    frame = LIVE_FRAMES.get(esp_id, {})
    if frame.get("image") is None:
        raise HTTPException(404, "Nema slike iz ove sesije.")
    if sequence != frame.get("sequence"):
        raise HTTPException(409, "Stigla je nova slika; osveži prikaz.")
    return Response(frame["image"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/update")
async def azuriraj_rucno(podatak: PodatakSaSata):
    if podatak.sat not in ["sat_1", "sat_2"]:
        raise HTTPException(status_code=400, detail="Koristi 'sat_1' ili 'sat_2'.")

    insert_measurement("manual", podatak.sat, podatak.vrednost, None, None, None, "manual", True)
    return {"status": "uspeh"}


@app.get("/api/podaci")
async def api_podaci():
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT sat, vrednost, vreme FROM merenja
            WHERE vrednost IS NOT NULL AND ocr_ok = 1
            ORDER BY id DESC LIMIT 100
        """).fetchall()
    finally:
        conn.close()
    istorija = {"sat_1": [], "sat_2": []}
    trenutno = {"sat_1": None, "sat_2": None}
    for row in reversed(rows):
        if row["sat"] in istorija:
            istorija[row["sat"]].append({"vreme": row["vreme"], "vrednost": row["vrednost"]})
    slike = {}
    now = time.monotonic()
    for device, cfg in METER_CONFIG.items():
        frame = LIVE_FRAMES.get(device, {})
        fresh = frame.get("received_at") is not None and now-frame["received_at"] <= 15
        value = frame.get("value") if fresh and frame.get("ocr_ok") else None
        trenutno[cfg["sat"]] = value
        slike[device] = {
            "original_slika": frame.get("original_slika"), "crop_slika": None,
            "processed_slika": None, "ocr_raw": frame.get("ocr_raw"),
            "ocr_ok": bool(value is not None), "vreme": frame.get("vreme"),
            "sequence": frame.get("sequence", 0), "latency_ms": frame.get("latency_ms"),
            "status": frame.get("status", "waiting") if fresh else "waiting_new_frame",
        }
    return {"trenutno": trenutno, "istorija": istorija, "slike": slike}


@app.get("/api/merenja")
async def api_merenja(limit: int = 50):
   
    limit = max(1, min(limit, 200))

    conn = get_conn()
    rows = conn.execute("""
        SELECT id, esp_id, sat, vrednost, vreme, ocr_ok, ocr_raw, original_slika, crop_slika, processed_slika
        FROM merenja
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return [dict(r) for r in rows]


@app.delete("/api/obrisi")
async def obrisi_bazu():
    async with AsyncExitStack() as stack:
        for lock in UPLOAD_LOCKS.values():
            await stack.enter_async_context(lock)
        result = await asyncio.to_thread(delete_saved_measurements)
        LIVE_FRAMES.clear()
        for device in METER_CONFIG:
            OCR_CONFIRMATION.invalidate(device)
        return result


def delete_saved_measurements():
    # 1. Obriši sva merenja iz baze
    conn = get_conn()

    try:
        conn.execute("DELETE FROM merenja")
        conn.commit()
    finally:
        conn.close()

    # 2. Obriši sve slike / OCR debug fajlove iz pictures foldera
    deleted_files = 0

    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            # Brišemo samo fajlove, ne folder
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    deleted_files += 1
                except Exception as e:
                    print(f"[BRISANJE] Ne mogu da obrišem {file_path}: {e}")

    except Exception as e:
        print(f"[BRISANJE] Greška pri čišćenju pictures foldera: {e}")

    print(
        f"[{current_time()}] Baza obrisana. "
        f"Obrisano fajlova: {deleted_files}"
    )

    return {
        "status": "baza i slike obrisane",
        "deleted_files": deleted_files
    }



@app.get("/api/debug/{esp_id}")
async def debug_paths(esp_id: str):
    if esp_id not in METER_CONFIG:
        raise HTTPException(status_code=404, detail="Nepoznat ESP ID.")

    names = {
        "normalized": f"latest_normalized_{esp_id}.jpg",
        "blue_mask": f"latest_blue_mask_{esp_id}.png",
        "blue_mask_strong": f"latest_blue_mask_strong_{esp_id}.png",
        "blue_mask_broad": f"latest_blue_mask_broad_{esp_id}.png",
        "screen": f"latest_screen_{esp_id}.jpg",
        "crop": f"latest_crop_{esp_id}.jpg",
        "processed": f"latest_processed_{esp_id}.png",
        "segments_0": f"latest_segments_{esp_id}_0.png",
        "segments_1": f"latest_segments_{esp_id}_1.png",
        "segments_2": f"latest_segments_{esp_id}_2.png",
    }

    return {
        "version": SERVER_VERSION,
        "esp_id": esp_id,
        "files": {
            key: (f"/pictures/{name}" if os.path.exists(os.path.join(UPLOAD_FOLDER, name)) else None)
            for key, name in names.items()
        }
    }


@app.get("/api/status")
async def status():
    return {
        "version": SERVER_VERSION,
        "server": "radi",
        "ocr_engine": "structural_7seg",
        "confirmation_required": 2,
        "ssocr_installed": shutil.which("ssocr") is not None,
        "tesseract_installed": shutil.which("tesseract") is not None,
        "pictures_folder": UPLOAD_FOLDER,
        "config": METER_CONFIG
    }
