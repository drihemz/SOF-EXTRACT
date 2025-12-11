import io
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from pdf2image import convert_from_bytes
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

# Regex for times & dates
TIME_REGEX = re.compile(r"(\d{1,2}[:h\.]\d{2})", re.IGNORECASE)
DATE_REGEX = re.compile(r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})")

# Settings
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_SECONDS = 260
OCR_DPI = 200
MIN_INK_RATIO = 0.001

# Paddle OCR engine (CPU)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False,
    show_log=False,
)

# Helpers ------------------------------------------------------------------------------------

def extract_times_and_dates(text: str):
    dates = DATE_REGEX.findall(text) or []
    times_raw = TIME_REGEX.findall(text) or []

    cleaned = []
    for t in times_raw:
        if "." in t and any(d.startswith(t) for d in dates):
            continue
        t2 = t.lower().replace("h", ":").replace(".", ":")
        parts = t2.split(":")
        if len(parts) != 2:
            continue
        try:
            h = int(parts[0]); m = int(parts[1])
        except:
            continue
        if 0 <= h <= 23 and 0 <= m <= 59:
            cleaned.append(f"{h:02d}:{m:02d}")
    return cleaned, dates

def compute_nonwhite_ratio(img: Image.Image) -> float:
    g = img.convert("L")
    hist = g.histogram()
    total = sum(hist)
    white = hist[-1]
    return (total - white) / max(total, 1)

def paddle_ocr_page(img: Image.Image, page_index: int):
    events, boxes = [], []

    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)

    result = ocr.ocr(arr, cls=True)
    if not result or not result[0]:
        return events, boxes

    # sort by vertical center
    rows = []
    for box, (txt, score) in result[0]:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        rows.append({
            "text": txt.strip(),
            "score": float(score),
            "bbox": {
                "x": float(x1), "y": float(y1),
                "width": float(x2 - x1), "height": float(y2 - y1)
            },
            "y_center": (y1 + y2) / 2,
        })

    rows.sort(key=lambda r: r["y_center"])

    events_out, boxes_out = [], []
    for i, row in enumerate(rows, 1):
        t = row["text"]
        times, dates = extract_times_and_dates(t)
        start = times[0] if len(times) >= 1 else None
        end   = times[1] if len(times) >= 2 else None

        ev = {
            "event": t,
            "notes": t,
            "start": start,
            "end": end,
            "dates": dates,
            "page": page_index,
            "line": i,
            "confidence": row["score"],
            "bbox": row["bbox"],
        }
        events_out.append(ev)

        boxes_out.append({
            "page": page_index,
            "line": i,
            "text": t,
            "bbox": row["bbox"],
            "confidence": row["score"],
        })

    return events_out, boxes_out

def ocr_pdf(content: bytes):
    start = time.monotonic()
    events, boxes, warnings = [], [], []

    try:
        pages = convert_from_bytes(content, dpi=OCR_DPI)
    except Exception as e:
        return [], [], [f"PDF conversion failed: {e}"], 0

    for idx, img in enumerate(pages, 1):
        if time.monotonic() - start > MAX_SECONDS:
            warnings.append("Time budget exceeded.")
            break

        if compute_nonwhite_ratio(img) < MIN_INK_RATIO:
            warnings.append(f"Page {idx} looks blank.")
            continue

        ev, bx = paddle_ocr_page(img, idx)
        events.extend(ev)
        boxes.extend(bx)

    return events, boxes, warnings, len(pages)

# API -----------------------------------------------------------------------------------------

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    is_pdf = file.filename.lower().endswith(".pdf")

    if is_pdf:
        events, boxes, warnings, total = ocr_pdf(raw)
    else:
        try:
            img = Image.open(io.BytesIO(raw))
        except:
            raise HTTPException(status_code=400, detail="Invalid image")
        events, boxes = paddle_ocr_page(img, 1)
        warnings, total = [], 1

    return JSONResponse({
        "events": events,
        "boxes": boxes,
        "warnings": warnings,
        "meta": {
            "sourcePages": total,
            "ocrDpi": OCR_DPI
        }
    })
