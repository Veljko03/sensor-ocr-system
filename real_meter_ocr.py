import cv2
import numpy as np
from pathlib import Path


# ============================================================
# REAL FESTO METER OCR
# KORAK 1: pronalazak LCD displeja + perspective correction
# ============================================================


def order_points(pts):
    """
    Redosled tačaka:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    rect = np.zeros((4, 2), dtype=np.float32)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_warp(image, points, out_w=500, out_h=500):
    """
    Ispravlja ekran koji je fotografisan pod uglom.
    """
    src = order_points(points)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (out_w, out_h)
    )

    return warped


def find_blue_screen(image):
    """
    Traži plavi LCD Festo uređaja.

    Ne oslanja se na tačnu poziciju ekrana u slici.
    """

    if image is None or image.size == 0:
        raise ValueError("Slika je prazna.")

    h, w = image.shape[:2]
    image_area = float(h * w)

    # --------------------------------------------------------
    # 1. HSV detekcija plave boje
    # --------------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([85, 35, 60], dtype=np.uint8)
    upper_blue = np.array([150, 255, 255], dtype=np.uint8)

    hsv_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # --------------------------------------------------------
    # 2. Dodatna blue-dominance detekcija
    #
    # Korisno kada je ekran presvetao / ispran.
    # --------------------------------------------------------

    b, g, r = cv2.split(image)

    b = b.astype(np.int16)
    g = g.astype(np.int16)
    r = r.astype(np.int16)

    blue_score = b - ((g + r) // 2)

    dominance_mask = np.where(
        blue_score > 18,
        255,
        0
    ).astype(np.uint8)

    # Kombinujemo obe maske
    mask = cv2.bitwise_or(
        hsv_mask,
        dominance_mask
    )

    # --------------------------------------------------------
    # 3. Čišćenje maske
    # --------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # 4. Tražimo moguće plave regione
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        area_fraction = area / image_area

        # Premalo = šum
        # Preveliko = skoro cela fotografija
        if area_fraction < 0.0008:
            continue

        if area_fraction > 0.50:
            continue

        rect = cv2.minAreaRect(contour)

        (_, _), (rw, rh), _ = rect

        if rw < 8 or rh < 8:
            continue

        short_side = min(rw, rh)
        long_side = max(rw, rh)

        aspect = short_side / max(long_side, 1.0)

        # Pravi Festo LCD je u nasem realnom datasetu skoro kvadratan
        # nakon detekcije: aspect >= 0.888.
        #
        # Hard-negative ceiling test je imao maksimum 0.784.
        # Koristimo konzervativnu granicu izmedju ta dva skupa.
        if aspect < 0.84:
            continue

        rect_area = rw * rh

        if rect_area <= 0:
            continue

        fill_ratio = area / rect_area

        if fill_ratio < 0.25:
            continue

        # Veći, puniji region ima veći score
        score = (
            area
            * (0.5 + fill_ratio)
        )

        candidates.append({
            "score": score,
            "contour": contour,
            "area_fraction": area_fraction,
            "fill_ratio": fill_ratio,
            "aspect": aspect
        })

    if not candidates:
        raise ValueError("Plavi LCD nije pronađen.")

    # Najbolji kandidat
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = candidates[0]

    rect = cv2.minAreaRect(
        best["contour"]
    )

    box = cv2.boxPoints(rect)

    # Perspective correction
    screen = perspective_warp(
        image,
        box,
        out_w=500,
        out_h=500
    )

    debug = {
        "area_fraction": round(
            float(best["area_fraction"]),
            5
        ),
        "fill_ratio": round(
            float(best["fill_ratio"]),
            3
        ),
        "aspect": round(
            float(best["aspect"]),
            3
        ),
        "candidates": len(candidates)
    }

    return screen, mask, box, debug
def extract_digit_region(screen):
    """
    Pronadji veliki red cifara na vec perspective-ispravljenom
    Festo LCD ekranu.

    Ne pretpostavlja fiksnu X poziciju cifara.
    Koristi cinjenicu da su cifre svetlije i manje "plave"
    od pozadine, kao i da su mnogo vece od tekstova Auto/bar.
    """

    if screen is None or screen.size == 0:
        raise ValueError("Prazan screen u extract_digit_region().")

    h, w = screen.shape[:2]

    # --------------------------------------------------------
    # 1. Napravi mapu piksela pogodnih za cifre
    # --------------------------------------------------------

    b, g, r = cv2.split(screen)

    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # Plava pozadina ima jak B kanal.
    # Svetle/zuckaste cifre imaju relativno jace R i G.
    digit_score = ((r + g) / 2.0) - (0.45 * b)

    digit_score = cv2.normalize(
        digit_score,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    digit_score = cv2.GaussianBlur(
        digit_score,
        (5, 5),
        0
    )

    # Otsu automatski bira prag
    _, mask = cv2.threshold(
        digit_score,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 2. Ne zanimaju nas sam vrh i samo dno LCD-a
    # --------------------------------------------------------

    search_mask = np.zeros_like(mask)

    y_start = int(0.10 * h)
    y_end = int(0.78 * h)

    search_mask[y_start:y_end, :] = mask[y_start:y_end, :]

    # --------------------------------------------------------
    # 3. Pronadji vertikalni pojas u kojem se nalaze velike cifre
    # --------------------------------------------------------

    row_counts = np.count_nonzero(search_mask, axis=1).astype(np.float32)

    smooth_kernel = np.ones(21, dtype=np.float32)

    row_smooth = np.convolve(
        row_counts,
        smooth_kernel,
        mode="same"
    )

    if row_smooth.max() <= 0:
        raise ValueError("Nema kandidata za cifre.")

    row_threshold = row_smooth.max() * 0.35

    active_rows = np.where(row_smooth >= row_threshold)[0]

    if len(active_rows) == 0:
        raise ValueError("Red cifara nije pronadjen.")

    # Uzimamo grupu aktivnih redova oko najjace tacke
    peak_y = int(np.argmax(row_smooth))

    groups = []
    start = active_rows[0]
    prev = active_rows[0]

    for y in active_rows[1:]:
        if y > prev + 1:
            groups.append((start, prev))
            start = y

        prev = y

    groups.append((start, prev))

    digit_group = None

    for y1, y2 in groups:
        if y1 <= peak_y <= y2:
            digit_group = (y1, y2)
            break

    if digit_group is None:
        raise ValueError("Nije moguce odrediti red cifara.")

    y1, y2 = digit_group

    # Prosiri region jer threshold uglavnom hvata srediste segmenata
    pad_y = int(0.08 * h)

    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)

    # --------------------------------------------------------
    # 4. Pronadji horizontalnu sirinu broja
    # --------------------------------------------------------

    number_band = mask[y1:y2 + 1, :]

    col_counts = np.count_nonzero(
        number_band,
        axis=0
    ).astype(np.float32)

    col_kernel = np.ones(11, dtype=np.float32)

    col_smooth = np.convolve(
        col_counts,
        col_kernel,
        mode="same"
    )

    if col_smooth.max() <= 0:
        raise ValueError("Nije pronadjena horizontalna zona cifara.")

    col_threshold = col_smooth.max() * 0.20

    active_cols = np.where(
        col_smooth >= col_threshold
    )[0]

    if len(active_cols) == 0:
        raise ValueError("Nisu pronadjene cifre po X osi.")

    x1 = int(active_cols[0])
    x2 = int(active_cols[-1])

    pad_x = int(0.03 * w)

    x1 = max(0, x1 - pad_x)
    x2 = min(w - 1, x2 + pad_x)

    # --------------------------------------------------------
    # 5. Iseci samo region sa velikim ciframa
    # --------------------------------------------------------

    digit_region = screen[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    digit_mask = mask[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    if digit_region.size == 0:
        raise ValueError("Dobijen je prazan digit region.")

    debug = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1 + 1,
        "height": y2 - y1 + 1,
        "screen_width": w,
        "screen_height": h
    }

    return digit_region, digit_mask, debug
def find_digit_boxes(digit_mask):
    """
    Pronadji pojedinacne cifre u izdvojenom digit region-u.

    Radi preko vertikalne projekcije, tako da ne zavisi od toga
    da li su svi segmenti jedne 7-segment cifre fizicki spojeni.

    Vraca:
        boxes = [(x1, y1, x2, y2), ...]
        clean_mask
        debug
    """

    if digit_mask is None or digit_mask.size == 0:
        raise ValueError("Prazan digit_mask.")

    mask = digit_mask.copy()

    h, w = mask.shape[:2]

    # --------------------------------------------------------
    # 1. Ukloni sitne povezane komponente
    # --------------------------------------------------------

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask)

    # Vrlo mali objekti su uglavnom tekst, tacke ili sum.
    min_area = max(
        20,
        int(h * w * 0.002)
    )

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean[labels == i] = 255

    # --------------------------------------------------------
    # 2. Nadji glavni vertikalni opseg velikih cifara
    # --------------------------------------------------------

    row_counts = np.count_nonzero(
        clean,
        axis=1
    ).astype(np.float32)

    if row_counts.max() <= 0:
        raise ValueError("Posle ciscenja nema belih piksela.")

    # Velike cifre zauzimaju veliki deo visine.
    active_rows = np.where(
        row_counts >= row_counts.max() * 0.12
    )[0]

    if len(active_rows) == 0:
        raise ValueError("Nije pronadjen vertikalni opseg cifara.")

    y1 = int(active_rows[0])
    y2 = int(active_rows[-1])

    # Malo prosirenje
    pad_y = max(2, int(0.03 * h))

    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)

    # Sve van glavnog reda cifara odbacujemo.
    cleaned_band = np.zeros_like(clean)
    cleaned_band[y1:y2 + 1, :] = clean[y1:y2 + 1, :]

    # --------------------------------------------------------
    # 3. Vertikalna projekcija
    #
    # Svaka cifra treba da napravi jednu horizontalnu grupu.
    # --------------------------------------------------------

    col_counts = np.count_nonzero(
        cleaned_band,
        axis=0
    ).astype(np.float32)

    # Blago peglanje da mali prekidi unutar jedne cifre
    # ne naprave vise grupa.
    kernel_size = max(5, int(w * 0.025))

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(
        kernel_size,
        dtype=np.float32
    )

    smooth = np.convolve(
        col_counts,
        kernel,
        mode="same"
    )

    if smooth.max() <= 0:
        raise ValueError("Nema kandidata po X osi.")

    threshold = smooth.max() * 0.10

    active_cols = smooth >= threshold

    # --------------------------------------------------------
    # 4. Pretvori aktivne kolone u grupe
    # --------------------------------------------------------

    groups = []
    start = None

    for x, active in enumerate(active_cols):

        if active and start is None:
            start = x

        elif not active and start is not None:
            groups.append((start, x - 1))
            start = None

    if start is not None:
        groups.append((start, w - 1))

    # --------------------------------------------------------
    # 5. Filtriraj premale grupe
    # --------------------------------------------------------

    boxes = []

    min_digit_width = max(
        8,
        int(w * 0.045)
    )

    for gx1, gx2 in groups:

        group_width = gx2 - gx1 + 1

        if group_width < min_digit_width:
            continue

        # Proveri koliko stvarnih belih piksela postoji
        # unutar kandidata.
        roi = cleaned_band[
            y1:y2 + 1,
            gx1:gx2 + 1
        ]

        white_pixels = np.count_nonzero(roi)

        if white_pixels < min_area:
            continue

        # Malo prosiri granice cifre
        pad_x = max(
            2,
            int(0.01 * w)
        )

        bx1 = max(0, gx1 - pad_x)
        bx2 = min(w - 1, gx2 + pad_x)

        boxes.append((
            int(bx1),
            int(y1),
            int(bx2),
            int(y2)
        ))

    if not boxes:
        raise ValueError("Nijedna cifra nije pronadjena.")

    debug = {
        "digit_count": len(boxes),
        "boxes": boxes,
        "y1": y1,
        "y2": y2,
        "min_area": min_area,
        "projection_threshold": round(
            float(threshold),
            2
        )
    }

    return boxes, cleaned_band, debug
def find_digit_boxes_v2(digit_mask):
    """
    Robustnija detekcija pojedinacnih 7-segment cifara.

    Problem koji resava:
    kod blur-a susedne cifre mogu da se "spoje" u jednu siroku grupu.

    Resenje:
    1. pronadji grube horizontalne grupe
    2. proceni tipicnu sirinu jedne cifre iz visine cifara
    3. prevelike grupe podeli na 2+ cifara
       trazenjem minimuma vertikalne projekcije
    """

    if digit_mask is None or digit_mask.size == 0:
        raise ValueError("Prazan digit_mask.")

    mask = digit_mask.copy()

    h, w = mask.shape[:2]

    # ========================================================
    # 1. Uklanjanje veoma sitnog suma
    # ========================================================

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask)

    min_area = max(
        15,
        int(h * w * 0.001)
    )

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area >= min_area:
            clean[labels == i] = 255

    if np.count_nonzero(clean) == 0:
        raise ValueError("Posle ciscenja nema kandidata.")

    # ========================================================
    # 2. Pronadji glavni red cifara
    # ========================================================

    row_counts = np.count_nonzero(
        clean,
        axis=1
    ).astype(np.float32)

    # Peglanje vertikalne projekcije
    row_kernel = np.ones(
        max(5, int(h * 0.05)),
        dtype=np.float32
    )

    row_smooth = np.convolve(
        row_counts,
        row_kernel,
        mode="same"
    )

    peak_y = int(np.argmax(row_smooth))

    # Od maksimuma idemo gore/dole dok signal
    # ne padne dovoljno.
    row_threshold = row_smooth[peak_y] * 0.12

    y1 = peak_y
    while y1 > 0 and row_smooth[y1] >= row_threshold:
        y1 -= 1

    y2 = peak_y
    while y2 < h - 1 and row_smooth[y2] >= row_threshold:
        y2 += 1

    pad_y = max(3, int(h * 0.04))

    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)

    band = np.zeros_like(clean)
    band[y1:y2 + 1, :] = clean[y1:y2 + 1, :]

    digit_height = y2 - y1 + 1

    # ========================================================
    # 3. Horizontalna projekcija
    # ========================================================

    col_counts = np.count_nonzero(
        band,
        axis=0
    ).astype(np.float32)

    # Samo blago peglanje.
    # Ne zelimo ponovo da spojimo susedne cifre.
    col_kernel_size = max(3, int(w * 0.012))

    if col_kernel_size % 2 == 0:
        col_kernel_size += 1

    col_smooth = np.convolve(
        col_counts,
        np.ones(col_kernel_size, dtype=np.float32),
        mode="same"
    )

    if col_smooth.max() <= 0:
        raise ValueError("Nema aktivnosti po X osi.")

    threshold = col_smooth.max() * 0.075

    active = col_smooth >= threshold

    # ========================================================
    # 4. Pocetne grube grupe
    # ========================================================

    rough_groups = []

    start = None

    for x, is_active in enumerate(active):

        if is_active and start is None:
            start = x

        elif not is_active and start is not None:
            rough_groups.append((start, x - 1))
            start = None

    if start is not None:
        rough_groups.append((start, w - 1))

    rough_groups = [
        (x1, x2)
        for x1, x2 in rough_groups
        if (x2 - x1 + 1) >= max(6, int(w * 0.025))
    ]

    if not rough_groups:
        raise ValueError("Nisu pronadjene grupe cifara.")

    # ========================================================
    # 5. Procena sirine JEDNE cifre
    #
    # 7-segment cifra je otprilike 40-50% svoje visine.
    # Na nasem Festo displeju oko 0.45 daje dobru polaznu tacku.
    # ========================================================

    expected_digit_width = max(
        12,
        int(digit_height * 0.56)
    )

    # ========================================================
    # 6. Podela prevelikih grupa
    # ========================================================

    split_groups = []

    # Za trazenje "doline" koristimo vrlo blago
    # peglanu projekciju.
    valley_kernel = np.ones(5, dtype=np.float32)

    valley_projection = np.convolve(
        col_counts,
        valley_kernel,
        mode="same"
    )

    for gx1, gx2 in rough_groups:

        group_width = gx2 - gx1 + 1

        # Koliko cifara verovatno postoji u grupi?

        estimated_digits = max(
            1,
            int(np.ceil(
                group_width / (expected_digit_width * 1.25)
            ))
        )

        # Ne dozvoljavamo sulude podele.
        estimated_digits = min(
            estimated_digits,
            5
        )

        if estimated_digits == 1:
            split_groups.append((gx1, gx2))
            continue

        cuts = []

        # Idealne tacke podele, npr. za dve cifre:
        # oko 50% sirine grupe.
        for part in range(1, estimated_digits):

            ideal = gx1 + (
                group_width * part / estimated_digits
            )

            search_radius = max(
                6,
                int(expected_digit_width * 0.22)
            )

            sx1 = max(
                gx1 + 3,
                int(ideal - search_radius)
            )

            sx2 = min(
                gx2 - 3,
                int(ideal + search_radius)
            )

            if sx2 <= sx1:
                cut = int(round(ideal))
            else:
                local = valley_projection[sx1:sx2 + 1]

                cut = sx1 + int(
                    np.argmin(local)
                )

            cuts.append(cut)

        boundaries = [gx1] + cuts + [gx2]

        for i in range(len(boundaries) - 1):

            bx1 = boundaries[i]

            # Da se susedne kutije ne preklapaju previse
            if i > 0:
                bx1 += 1

            bx2 = boundaries[i + 1]

            if bx2 - bx1 + 1 >= 8:
                split_groups.append(
                    (int(bx1), int(bx2))
                )

    # ========================================================
    # 7. Finalni bounding box za svaku cifru
    # ========================================================

    boxes = []

    pad_x = max(
        2,
        int(w * 0.008)
    )

    for gx1, gx2 in split_groups:

        roi = band[
            y1:y2 + 1,
            gx1:gx2 + 1
        ]

        if np.count_nonzero(roi) < min_area:
            continue

        bx1 = max(0, gx1 - pad_x)
        bx2 = min(w - 1, gx2 + pad_x)

        boxes.append((
            int(bx1),
            int(y1),
            int(bx2),
            int(y2)
        ))

    boxes.sort(
        key=lambda b: b[0]
    )

    if not boxes:
        raise ValueError("Nijedna cifra nije pronadjena.")

    debug = {
        "digit_count": len(boxes),
        "boxes": boxes,
        "rough_groups": rough_groups,
        "split_groups": split_groups,
        "digit_height": digit_height,
        "expected_digit_width": expected_digit_width,
        "y1": y1,
        "y2": y2
    }

    return boxes, band, debug
def extract_digit_region_v3(screen):
    """
    Izdvaja zonu velikih cifara sa pravog Festo displeja.

    Glavna razlika u odnosu na prethodnu verziju:
    - ne trazi najsvetliji deo preko skoro celog LCD-a
    - pretrazuje samo centralni pojas gde se nalaze velike cifre
    - radi isto bez obzira da li broj ima 3 ili 4 cifre
    """

    if screen is None or screen.size == 0:
        raise ValueError("Prazan screen u extract_digit_region_v3().")

    h, w = screen.shape[:2]

    # ========================================================
    # 1. OGRANICI PRETRAGU NA ZONU VELIKIH CIFARA
    #
    # Gornji deo sadrzi statusne simbole / Auto / Out...
    # Donji deo sadrzi "bar".
    #
    # Perspective warp nam je ekran standardizovao, pa je
    # srednji pojas stabilan bez obzira na ugao kamere.
    # ========================================================

    search_y1 = int(0.24 * h)
    search_y2 = int(0.69 * h)

    search_region = screen[
        search_y1:search_y2,
        :
    ]

    # ========================================================
    # 2. DETEKCIJA SVETLIH / ZUCKASTIH CIFARA
    # ========================================================

    b, g, r = cv2.split(search_region)

    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # Svetle cifre imaju relativno jak R i G,
    # dok plava pozadina ima dominantan B.
    digit_score = (
        0.50 * r +
        0.50 * g -
        0.35 * b
    )

    digit_score = cv2.normalize(
        digit_score,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    digit_score = cv2.GaussianBlur(
        digit_score,
        (5, 5),
        0
    )

    _, mask = cv2.threshold(
        digit_score,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ========================================================
    # 3. UKLONI SITAN SUM
    # ========================================================

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask)

    min_area = max(
        15,
        int(mask.size * 0.0008)
    )

    for i in range(1, n):

        area = int(stats[i, cv2.CC_STAT_AREA])
        comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Velike cifre imaju relativno visoke komponente.
        # Ovo uklanja vecinu sitnog teksta.
        if area >= min_area and comp_h >= int(mask.shape[0] * 0.12):
            clean[labels == i] = 255

    if np.count_nonzero(clean) == 0:
        raise ValueError("Nema kandidata za velike cifre.")

    # ========================================================
    # 4. PRONADJI GLAVNI VERTIKALNI RED CIFARA
    # ========================================================

    row_counts = np.count_nonzero(
        clean,
        axis=1
    ).astype(np.float32)

    kernel_size = max(
        7,
        int(mask.shape[0] * 0.07)
    )

    row_smooth = np.convolve(
        row_counts,
        np.ones(kernel_size, dtype=np.float32),
        mode="same"
    )

    peak_y = int(np.argmax(row_smooth))

    threshold = row_smooth[peak_y] * 0.18

    y1_local = peak_y

    while (
        y1_local > 0 and
        row_smooth[y1_local] >= threshold
    ):
        y1_local -= 1

    y2_local = peak_y

    while (
        y2_local < len(row_smooth) - 1 and
        row_smooth[y2_local] >= threshold
    ):
        y2_local += 1

    pad_y = int(0.05 * h)

    y1_local = max(
        0,
        y1_local - pad_y
    )

    y2_local = min(
        search_region.shape[0] - 1,
        y2_local + pad_y
    )

    # ========================================================
    # 5. PRONADJI HORIZONTALNI OPSEG CELOG BROJA
    #
    # Ovde NE pretpostavljamo 3 cifre.
    # Moze da bude:
    # 585  -> 5.85
    # 2244 -> 22.44
    # ========================================================

    digit_band = clean[
        y1_local:y2_local + 1,
        :
    ]

    col_counts = np.count_nonzero(
        digit_band,
        axis=0
    ).astype(np.float32)

    col_smooth = np.convolve(
        col_counts,
        np.ones(7, dtype=np.float32),
        mode="same"
    )

    if col_smooth.max() <= 0:
        raise ValueError("Nema horizontalnog regiona cifara.")

    col_threshold = col_smooth.max() * 0.08

    active_cols = np.where(
        col_smooth >= col_threshold
    )[0]

    if len(active_cols) == 0:
        raise ValueError("Nisu pronadjene cifre po X osi.")

    x1 = int(active_cols[0])
    x2 = int(active_cols[-1])

    pad_x = int(0.025 * w)

    x1 = max(0, x1 - pad_x)
    x2 = min(w - 1, x2 + pad_x)

    # Lokalni Y pretvaramo nazad u koordinate celog screen-a.
    y1 = search_y1 + y1_local
    y2 = search_y1 + y2_local

    # ========================================================
    # 6. FINALNI REGION
    # ========================================================

    digit_region = screen[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    digit_mask = clean[
        y1_local:y2_local + 1,
        x1:x2 + 1
    ]

    if digit_region.size == 0:
        raise ValueError("Dobijen je prazan digit region.")

    debug = {
        "search_y1": search_y1,
        "search_y2": search_y2,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1 + 1,
        "height": y2 - y1 + 1
    }

    return digit_region, digit_mask, debug
def split_three_digit_slots(digit_region):
    """
    Deli izdvojeni broj na tacno 3 digit slota.

    Sistem je ogranicen na vrednosti 0.00 - 9.99,
    pa displej uvek predstavlja tacno tri cifre:

        023 -> 0.23
        161 -> 1.61
        732 -> 7.32

    Ne pokusavamo da zakljucimo sirinu cifre iz njenog oblika.
    Ovo je posebno vazno za cifru 1, koja je mnogo uza.
    """

    if digit_region is None or digit_region.size == 0:
        raise ValueError("Prazan digit_region.")

    h, w = digit_region.shape[:2]

    # Tri jednaka logicka slota.
    boundaries = [
        0,
        round(w / 3),
        round(2 * w / 3),
        w
    ]

    slots = []

    # Malo preklapanje da segment na granici ne bude odsecen.
    overlap = 0

    for i in range(3):
        x1 = boundaries[i]
        x2 = boundaries[i + 1]

        if i > 0:
            x1 = max(0, x1 - overlap)

        if i < 2:
            x2 = min(w, x2 + overlap)

        slot = digit_region[:, x1:x2]

        if slot.size == 0:
            raise ValueError(f"Digit slot {i + 1} je prazan.")

        slots.append(slot)

    return slots
def slot_to_binary_mask(slot_bgr):
    """
    Iz jednog digit slota pravi binarnu masku cifre.

    Ideja:
    - cifra je svetlija/zuckastija od plave pozadine
    - radimo threshold samo unutar jednog slota
    - rezultat je bela cifra na crnoj pozadini
    """

    if slot_bgr is None or slot_bgr.size == 0:
        raise ValueError("Prazan slot_bgr.")

    b, g, r = cv2.split(slot_bgr)

    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    score = (
        0.50 * r +
        0.50 * g -
        0.35 * b
    )

    score = cv2.normalize(
        score,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    score = cv2.GaussianBlur(score, (5, 5), 0)

    _, mask = cv2.threshold(
        score,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ukloni sitan sum
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask)

    min_area = max(10, int(mask.size * 0.01))

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == i] = 255

    if np.count_nonzero(clean) == 0:
        return mask

    return clean
def extract_fixed_three_slots(screen):
    """
    Izdvaja tacno 3 digit slota sa normalizovanog Festo LCD-a.

    Sistem meri samo vrednosti 0.00 - 9.99,
    pa su na displeju uvek tacno 3 cifre.
    """

    if screen is None or screen.size == 0:
        raise ValueError("Prazan screen.")

    h, w = screen.shape[:2]

    x_edges = [
        int(0.34 * w),
        int(0.56 * w),
        int(0.77 * w),
        int(0.98 * w)
    ]

    y1 = int(0.24 * h)
    y2 = int(0.58 * h)

    slots = []

    for i in range(3):
        slot = screen[
            y1:y2,
            x_edges[i]:x_edges[i + 1]
        ]

        if slot.size == 0:
            raise ValueError(f"Slot {i + 1} je prazan.")

        slots.append(slot)

    return slots
REAL_SEVEN_SEGMENT_PATTERNS = {
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
}


def decode_fixed_slot(slot_mask):
    """
    Prepoznaje jednu 7-segment cifru iz jednog fiksnog Festo slota.

    Segmenti:
          a
        f   b
          g
        e   c
          d

    Vraca:
        digit, confidence, debug
    """

    if slot_mask is None or slot_mask.size == 0:
        raise ValueError("Prazan slot_mask.")

    if len(slot_mask.shape) == 3:
        slot_mask = cv2.cvtColor(
            slot_mask,
            cv2.COLOR_BGR2GRAY
        )

    # Sigurno binarno 0/255
    mask = np.where(
        slot_mask > 127,
        255,
        0
    ).astype(np.uint8)

    h, w = mask.shape[:2]

    # --------------------------------------------------------
    # Poseban slucaj za cifru 1.
    #
    # Cifra 1 je mnogo uza od svih ostalih cifara.
    # To nam je ranije pravilo problem, ali ovde je zapravo
    # veoma korisna karakteristika.
    # --------------------------------------------------------

    points = cv2.findNonZero(mask)

    if points is None:
        return None, 0.0, {
            "reason": "nema belih piksela"
        }

    x, y, bw, bh = cv2.boundingRect(points)

    active_width_ratio = bw / max(1.0, float(w))

    if active_width_ratio < 0.38:
        return "1", 0.98, {
            "reason": "uska cifra -> 1",
            "active_width_ratio": round(
                active_width_ratio,
                3
            )
        }

    # --------------------------------------------------------
    # Nominalne zone sedam segmenata.
    #
    # Zone su malo sire, a dole dodatno probamo mala pomeranja
    # kako sitna odstupanja kamere ne bi bila problem.
    # --------------------------------------------------------

    rois = [
        (0.20, 0.02, 0.80, 0.17),  # a
        (0.68, 0.12, 0.96, 0.46),  # b
        (0.68, 0.54, 0.96, 0.88),  # c
        (0.20, 0.83, 0.80, 0.98),  # d
        (0.04, 0.54, 0.32, 0.88),  # e
        (0.04, 0.12, 0.32, 0.46),  # f
        (0.20, 0.42, 0.80, 0.58),  # g
    ]

    best = None

    # Mala pretraga oko nominalne pozicije.
    for dx_ratio in (-0.04, -0.02, 0.0, 0.02, 0.04):
        for dy_ratio in (-0.025, 0.0, 0.025):

            occupancies = []

            for rx1, ry1, rx2, ry2 in rois:

                x1 = int((rx1 + dx_ratio) * w)
                x2 = int((rx2 + dx_ratio) * w)

                y1 = int((ry1 + dy_ratio) * h)
                y2 = int((ry2 + dy_ratio) * h)

                x1 = max(0, min(w, x1))
                x2 = max(0, min(w, x2))

                y1 = max(0, min(h, y1))
                y2 = max(0, min(h, y2))

                roi = mask[y1:y2, x1:x2]

                if roi.size == 0:
                    occupancy = 0.0
                else:
                    occupancy = float(
                        np.mean(roi > 0)
                    )

                occupancies.append(occupancy)

            # ------------------------------------------------
            # Uporedi izmerene segmente sa svih 10 obrazaca.
            # ------------------------------------------------

            for digit, pattern in REAL_SEVEN_SEGMENT_PATTERNS.items():

                # 1 smo vec posebno resili.
                if digit == "1":
                    continue

                cost = 0.0

                for observed, expected in zip(
                    occupancies,
                    pattern
                ):
                    cost += (
                        observed - float(expected)
                    ) ** 2

                cost /= 7.0

                candidate = {
                    "digit": digit,
                    "cost": float(cost),
                    "occupancies": occupancies,
                    "dx_ratio": dx_ratio,
                    "dy_ratio": dy_ratio
                }

                if (
                    best is None
                    or candidate["cost"] < best["cost"]
                ):
                    best = candidate

    if best is None:
        return None, 0.0, {
            "reason": "nema kandidata"
        }

    confidence = max(
        0.0,
        min(
            1.0,
            1.0 - best["cost"]
        )
    )

    debug = {
        "digit": best["digit"],
        "cost": round(best["cost"], 4),
        "confidence": round(confidence, 4),
        "occupancies": [
            round(v, 3)
            for v in best["occupancies"]
        ],
        "dx_ratio": best["dx_ratio"],
        "dy_ratio": best["dy_ratio"],
        "active_width_ratio": round(
            active_width_ratio,
            3
        )
    }

    return best["digit"], confidence, debug
def decode_fixed_slot_v2(slot_mask):
    """
    V2 decoder za jednu Festo 7-segment cifru.

    Glavna promena:
    occupancy prvo pretvaramo u ON/OFF segment,
    pa prvenstveno poredimo 7-segment obrazac.

    Ovo je robustnije na blur i slabije osvetljene segmente.
    """

    if slot_mask is None or slot_mask.size == 0:
        raise ValueError("Prazan slot_mask.")

    if len(slot_mask.shape) == 3:
        slot_mask = cv2.cvtColor(
            slot_mask,
            cv2.COLOR_BGR2GRAY
        )

    mask = np.where(
        slot_mask > 127,
        255,
        0
    ).astype(np.uint8)

    h, w = mask.shape[:2]

    points = cv2.findNonZero(mask)

    if points is None:
        return None, 0.0, {
            "reason": "nema belih piksela"
        }

    x, y, bw, bh = cv2.boundingRect(points)

    active_width_ratio = bw / max(1.0, float(w))

    # Cifra 1 je karakteristicno vrlo uska.
    if active_width_ratio < 0.55:
        return "1", 0.98, {
            "reason": "uska cifra -> 1",
            "active_width_ratio": round(
                active_width_ratio,
                3
            )
        }

    rois = [
        (0.20, 0.02, 0.80, 0.17),  # a
        (0.68, 0.12, 0.96, 0.46),  # b
        (0.68, 0.54, 0.96, 0.88),  # c
        (0.20, 0.83, 0.80, 0.98),  # d
        (0.04, 0.54, 0.32, 0.88),  # e
        (0.04, 0.12, 0.32, 0.46),  # f
        (0.20, 0.42, 0.80, 0.58),  # g
    ]

    SEGMENT_THRESHOLD = 0.25

    best = None

    for dx_ratio in (-0.04, -0.02, 0.0, 0.02, 0.04):
        for dy_ratio in (-0.025, 0.0, 0.025):

            occupancies = []

            for rx1, ry1, rx2, ry2 in rois:

                x1 = int((rx1 + dx_ratio) * w)
                x2 = int((rx2 + dx_ratio) * w)

                y1 = int((ry1 + dy_ratio) * h)
                y2 = int((ry2 + dy_ratio) * h)

                x1 = max(0, min(w, x1))
                x2 = max(0, min(w, x2))

                y1 = max(0, min(h, y1))
                y2 = max(0, min(h, y2))

                roi = mask[y1:y2, x1:x2]

                if roi.size == 0:
                    occupancy = 0.0
                else:
                    occupancy = float(
                        np.mean(roi > 0)
                    )

                occupancies.append(occupancy)

            observed_pattern = tuple(
                1 if value >= SEGMENT_THRESHOLD else 0
                for value in occupancies
            )

            for digit, expected_pattern in REAL_SEVEN_SEGMENT_PATTERNS.items():

                if digit == "1":
                    continue

                # Najvaznije: koliko ON/OFF segmenata se razlikuje.
                mismatches = sum(
                    observed != expected
                    for observed, expected in zip(
                        observed_pattern,
                        expected_pattern
                    )
                )

                # Tie-breaker ako dve cifre imaju isti broj gresaka.
                soft_cost = sum(
                    (observed - float(expected)) ** 2
                    for observed, expected in zip(
                        occupancies,
                        expected_pattern
                    )
                ) / 7.0

                score = (
                    mismatches * 10.0
                    + soft_cost
                )

                candidate = {
                    "digit": digit,
                    "score": float(score),
                    "mismatches": int(mismatches),
                    "soft_cost": float(soft_cost),
                    "occupancies": occupancies,
                    "pattern": observed_pattern,
                    "dx_ratio": dx_ratio,
                    "dy_ratio": dy_ratio
                }

                if (
                    best is None
                    or candidate["score"] < best["score"]
                ):
                    best = candidate

    if best is None:
        return None, 0.0, {
            "reason": "nema kandidata"
        }

    # 0 mismatch = veoma jaka sigurnost.
    confidence = max(
        0.0,
        min(
            1.0,
            1.0
            - 0.18 * best["mismatches"]
            - 0.25 * best["soft_cost"]
        )
    )

    debug = {
        "digit": best["digit"],
        "mismatches": best["mismatches"],
        "soft_cost": round(best["soft_cost"], 4),
        "confidence": round(confidence, 4),
        "pattern": best["pattern"],
        "occupancies": [
            round(v, 3)
            for v in best["occupancies"]
        ],
        "dx_ratio": best["dx_ratio"],
        "dy_ratio": best["dy_ratio"],
        "active_width_ratio": round(
            active_width_ratio,
            3
        )
    }

    return best["digit"], confidence, debug
def read_three_digit_value(screen, confidence_threshold=0.45):
    """
    Cita vrednost sa Festo displeja.

    Pretpostavka sistema:
        vrednost je uvek 0.00 - 9.99
        i zato displej uvek sadrzi tacno 3 cifre.

    Primer:
        023 -> 0.23
        161 -> 1.61
        706 -> 7.06

    Ako bilo koja cifra nema dovoljnu sigurnost,
    celo ocitavanje se odbija.
    """

    slots = extract_fixed_three_slots(screen)

    digits = []
    confidences = []
    digit_debug = []

    for index, slot in enumerate(slots, start=1):

        mask = slot_to_binary_mask(slot)

        digit, confidence, debug = decode_fixed_slot_v2(mask)

        digit_debug.append({
            "slot": index,
            "digit": digit,
            "confidence": round(float(confidence), 4),
            "debug": debug
        })

        if digit is None:
            return None, {
                "ok": False,
                "reason": f"slot {index}: cifra nije prepoznata",
                "digits": digit_debug
            }

        if confidence < confidence_threshold:
            return None, {
                "ok": False,
                "reason": (
                    f"slot {index}: confidence {confidence:.3f} "
                    f"< {confidence_threshold:.3f}"
                ),
                "digits": digit_debug
            }

        digits.append(digit)
        confidences.append(float(confidence))

    digit_string = "".join(digits)

    value = int(digit_string) / 100.0

    return value, {
        "ok": True,
        "digit_string": digit_string,
        "value": value,
        "min_confidence": round(min(confidences), 4),
        "avg_confidence": round(
            sum(confidences) / len(confidences),
            4
        ),
        "digits": digit_debug
    }
def read_real_meter_image(image, confidence_threshold=0.45):
    """
    Glavna funkcija za citanje realnog Festo meraca
    iz originalne fotografije.

    Vraca:
        value, debug

    value:
        float ako je ocitavanje pouzdano
        None ako LCD nije pronadjen ili OCR nije dovoljno siguran
    """

    if image is None or image.size == 0:
        return None, {
            "ok": False,
            "reason": "prazna slika"
        }

    try:
        screen, _, _, _ = find_blue_screen(image)
    except ValueError as e:
        return None, {
            "ok": False,
            "reason": str(e),
            "stage": "screen_detection"
        }

    try:
        value, reading_debug = read_three_digit_value(
            screen,
            confidence_threshold=confidence_threshold
        )
    except Exception as e:
        return None, {
            "ok": False,
            "reason": str(e),
            "stage": "digit_reading"
        }

    return value, {
        "ok": value is not None,
        "stage": "complete",
        "reading": reading_debug
    }
