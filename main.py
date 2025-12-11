import io
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from paddleocr import PaddleOCR

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Regex for time / date extraction (SOF-friendly)
# ---------------------------------------------------------------------------

TIME_REGEX = re.compile(r"(\d{1,2}[:h\.]\d{2})", re.IGNORECASE)
DATE_REGEX = re.compile(r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})")

# ---------------------------------------------------------------------------
# Tunables (env)
# ---------------------------------------------------------------------------

MAX_FILE_BYTES = int(os.environ.get("SOF_MAX_FILE_BYTES", str(40 * 1024 * 1024)))  # 40MB guard
MAX_SECONDS = int(os.environ.get("SOF_MAX_SECONDS", "260"))  # global soft guard
MAX_PDF_PAGES = int(os.environ.get("SOF_MAX_PDF_PAGES", "0"))  # 0 => all pages
OCR_DPI = int(os.environ.get("SOF_OCR_DPI", "200"))  # render DPI
MIN_INK_RATIO = float(os.environ.get("SOF_MIN_INK_RATIO", "0.001"))  # blank page cutoff

# PaddleOCR config
OCR_LANG = os.environ.get("SOF_OCR_LANG", "en")

# Instantiate PaddleOCR once (CPU only)
# This will download models on first run (Render outbound is allowed).
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang=OCR_LANG,
    use_gpu=False,
    show_log=False,
)


# ---------------------------------------------------------------------------
# Time / date helpers
# ---------------------------------------------------------------------------

def _sanitize_time(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    parts = val.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def extract_times_and_dates(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract normalized times (HH:MM) and raw date strings from a line of text.

    - Detects dates first (dd/mm/yy, dd-mm-yy, dd.mm.yy).
    - Skips time candidates that are clearly part of a date substring (e.g., 28.07 in 28.07.24).
    - Validates hour/minute ranges and drops impossible times.
    """
    dates = DATE_REGEX.findall(text) or []
    times_raw = TIME_REGEX.findall(text) or []

    cleaned_times: List[str] = []
    for t in times_raw:
        # If this candidate is a prefix of a date like "28.07.24", treat as date, not time
        if "." in t and any(d.startswith(t) for d in dates):
            continue
        t_norm = t.lower().replace("h", ":")
        if "." in t_norm:
            t_norm = t_norm.replace(".", ":")
        parts = t_norm.split(":")
        if len(parts) != 2:
            continue
        try:
            h = int(parts[0])
            m = int(parts[1])
        except ValueError:
            continue
        if not (0 <= h <= 24 and 0 <= m < 60):
            continue
        cleaned_times.append(f"{h:02d}:{m:02d}")

    seen = set()
    unique_times: List[str] = []
    for t in cleaned_times:
        if t not in seen:
            seen.add(t)
            unique_times.append(t)

    return unique_times, dates


# ---------------------------------------------------------------------------
# Simple ink / blank-page heuristic
# ---------------------------------------------------------------------------

def compute_nonwhite_ratio(img: Image.Image) -> float:
    """
    img: grayscale or RGB PIL image.
    Returns fraction of non-white-ish pixels, used to skip blanks.
    """
    if img.mode != "L":
        gray = img.convert("L")
    else:
        gray = img
    hist = gray.histogram()
    if not hist:
        return 0.0
    total = sum(hist)
    white = hist[-1] if len(hist) >= 256 else 0
    non_white = total - white
    return non_white / float(total or 1)


# ---------------------------------------------------------------------------
# Core PaddleOCR wrapper
# ---------------------------------------------------------------------------

def paddle_ocr_page(img: Image.Image, page_index: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Run PaddleOCR on a single page image and convert results into events + boxes.
    """
    events: List[Dict] = []
    boxes: List[Dict] = []

    # PaddleOCR works with numpy arrays (H, W, 3) BGR/RGB.
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_np = np.array(img)

    result = paddle_ocr.ocr(img_np, cls=True)

    # result is a list of [ [ (box, (text, score)), ... ] ] per image
    # We'll take the first image's results (we pass one page at a time)
    if not result or not result[0]:
        return events, boxes

    # Sort by vertical position to get a stable line order
    lines = []
    for line in result[0]:
        box, (txt, score) = line
        # box: 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        y_center = (y1 + y2) / 2.0
        lines.append({
            "text": txt.strip(),
            "score": float(score),
            "bbox": {"x": float(x1), "y": float(y1), "width": float(x2 - x1), "height": float(y2 - y1)},
            "y_center": float(y_center),
        })

    lines.sort(key=lambda l: l["y_center"])

    events_out: List[Dict] = []
    boxes_out: List[Dict] = []

    for idx, line in enumerate(lines, start=1):
        text = line["text"]
        if not text:
            continue
        times, dates = extract_times_and_dates(text)
        start = times[0] if len(times) >= 1 else None
        end = times[1] if len(times) >= 2 else None

        ev = {
            "event": text,
            "notes": text,
            "start": start,
            "end": end,
            "dates": list(dict.fromkeys(dates)),
            "ratePercent": None,
            "behavior": None,
            "page": page_index,
            "line": idx,
            "confidence": line["score"],
            "bbox": line["bbox"],
        }
        events_out.append(ev)
        boxes_out.append(
            {
                "page": page_index,
                "line": idx,
                "text": text,
                "bbox": line["bbox"],
                "confidence": line["score"],
            }
        )

    return events_out, boxes_out


# ---------------------------------------------------------------------------
# PDF pipeline (PyMuPDF + PaddleOCR)
# ---------------------------------------------------------------------------

def ocr_pdf_with_paddle(content: bytes) -> Tuple[List[Dict], List[Dict], List[str], int]:
    start = time.monotonic()
    warnings: List[str] = []
    events: List[Dict] = []
    boxes: List[Dict] = []

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        return [], [], [f"Failed to open PDF: {e}"], 0

    total_pages = doc.page_count
    limit = MAX_PDF_PAGES if MAX_PDF_PAGES > 0 else total_pages
    limit = min(limit, total_pages)

    for page_idx in range(limit):
        if time.monotonic() - start > MAX_SECONDS:
            warnings.append(f"OCR stopped early after reaching time budget ({MAX_SECONDS}s).")
            break

        page = doc.load_page(page_idx)

        # Render page to image
        zoom = OCR_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Skip near-blank pages
        if compute_nonwhite_ratio(img) < MIN_INK_RATIO:
            warnings.append(f"Page {page_idx + 1} skipped (looks blank).")
            continue

        page_events, page_boxes = paddle_ocr_page(img, page_index=page_idx + 1)
        events.extend(page_events)
        boxes.extend(page_boxes)

    doc.close()
    return events, boxes, warnings, total_pages


# ---------------------------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------------------------

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    req_start = time.monotonic()
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large for OCR (>{MAX_FILE_BYTES // (1024*1024)} MB). Please trim pages or reduce size.",
        )

    filename = (file.filename or "").lower()
    is_pdf = filename.endswith(".pdf")

    if is_pdf:
        events, boxes, ocr_warnings, page_count = ocr_pdf_with_paddle(content)
    else:
        # Treat as a single image
        try:
            img = Image.open(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {e}") from e

        if compute_nonwhite_ratio(img) < MIN_INK_RATIO:
            ocr_warnings = ["Image looks blank."]
            events, boxes = [], []
        else:
            events, boxes = paddle_ocr_page(img, page_index=1)
            ocr_warnings = []
        page_count = 1

    duration_ms = int((time.monotonic() - req_start) * 1000)

    return JSONResponse(
        {
            "events": events,
            "boxes": boxes,
            "warnings": ocr_warnings,
            "meta": {
                "sourcePages": page_count,
                "durationMs": duration_ms,
                "maxSeconds": MAX_SECONDS,
                "ocrDpi": OCR_DPI,
            },
        }
    )
