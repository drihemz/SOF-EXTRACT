import io
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import pytesseract

app = FastAPI()

# CORS: tighten origins to your frontend domains if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Regex for time / date extraction
# ---------------------------------------------------------------------------

TIME_REGEX = re.compile(r"(\d{1,2}[:h\.]\d{2})", re.IGNORECASE)
DATE_REGEX = re.compile(r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})")

# ---------------------------------------------------------------------------
# Tunables (env) – keep these simple to start
# ---------------------------------------------------------------------------

# SOF_MAX_PDF_PAGES=0 → process all pages
MAX_PDF_PAGES = int(os.environ.get("SOF_MAX_PDF_PAGES", "0"))

# Render at moderate DPI – good trade-off for your SOFs
BASE_DPI = int(os.environ.get("SOF_OCR_DPI", "180"))

# Guards
MAX_FILE_BYTES = int(os.environ.get("SOF_MAX_FILE_BYTES", str(40 * 1024 * 1024)))  # 40MB guard
MAX_SECONDS = int(os.environ.get("SOF_MAX_SECONDS", "240"))  # file-level soft guard

# treat page as text-rich if text layer has at least this many chars
MIN_TEXT_CHARS = int(os.environ.get("SOF_MIN_TEXT_CHARS", "150"))

# Tesseract config; psm 6 works well for these SOFs
BASE_TESS_CONFIG = os.environ.get(
    "SOF_TESS_CONFIG",
    "--oem 1 --psm 6 -c preserve_interword_spaces=1"
)

# No per-call Tesseract timeout by default; rely on MAX_SECONDS instead
TESSERACT_TIMEOUT = int(os.environ.get("SOF_TESS_TIMEOUT", "0"))  # 0 = no timeout


# ---------------------------------------------------------------------------
# Time / date extraction helpers
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
# Text-layer extraction (for digital PDFs)
# ---------------------------------------------------------------------------

def extract_text_layer_events(page: fitz.Page) -> List[Dict]:
    """Use the PDF text layer when present to avoid OCR cost and noise."""
    events: List[Dict] = []
    try:
        raw = page.get_text("rawdict") or {}
    except Exception:
        return events

    blocks = raw.get("blocks", [])
    line_counter = 0
    for block in blocks:
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            times, dates = extract_times_and_dates(text)
            start = times[0] if times else None
            end = times[1] if len(times) > 1 else None
            x1, y1, x2, y2 = line.get("bbox", (0, 0, 0, 0))
            bbox = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
            line_counter += 1
            events.append(
                {
                    "event": text,
                    "notes": text,
                    "start": start,
                    "end": end,
                    "dates": list(dict.fromkeys(dates)),
                    "page": page.number + 1,
                    "line": line_counter,
                    "confidence": 0.99,
                    "bbox": bbox,
                }
            )
    return events


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def parse_line_groups(ocr: Dict) -> List[Dict]:
    grouped: Dict[Tuple[int, int, int], List[int]] = {}
    n = len(ocr["text"])
    for i in range(n):
        text = ocr["text"][i]
        if not text or text.strip() == "":
            continue
        key = (ocr["block_num"][i], ocr["par_num"][i], ocr["line_num"][i])
        grouped.setdefault(key, []).append(i)

    lines = []
    for (b, p, l), idxs in grouped.items():
        idxs = sorted(idxs)
        words = []
        x1 = y1 = 10**9
        x2 = y2 = -10**9
        confs = []
        for i in idxs:
            words.append(ocr["text"][i])
            x, y, w, h = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x + w)
            y2 = max(y2, y + h)
            try:
                c = float(ocr["conf"][i])
                if c >= 0:
                    confs.append(c)
            except Exception:
                pass
        text = " ".join(words).strip()
        if not text:
            continue
        bbox = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
        avg_conf = sum(confs) / len(confs) if confs else None
        lines.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": (avg_conf / 100) if avg_conf is not None else None,
                "block": b,
                "paragraph": p,
                "line": l,
            }
        )
    return lines


def ocr_images(
    images: List[Image.Image],
    config: str = "--oem 1 --psm 6",
    base_page_index: int = 1,
) -> Tuple[List[Dict], List[Dict]]:
    events: List[Dict] = []
    boxes: List[Dict] = []
    seen_lines = set()

    for offset, img in enumerate(images):
        page_idx = base_page_index + offset

        # Build kwargs for pytesseract; only use timeout if >0
        kwargs = {
            "image": img,
            "output_type": pytesseract.Output.DICT,
            "config": config,
            "lang": "eng",
        }
        if TESSERACT_TIMEOUT > 0:
            kwargs["timeout"] = TESSERACT_TIMEOUT

        try:
            data = pytesseract.image_to_data(**kwargs)
        except RuntimeError as e:
            # Tesseract hung / exceeded timeout
            boxes.append(
                {
                    "page": page_idx,
                    "line": 0,
                    "text": f"OCR timeout on page {page_idx}: {e}",
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "confidence": 0.0,
                }
            )
            continue

        lines = parse_line_groups(data)
        for line_idx, line in enumerate(lines, start=1):
            clean = line["text"].strip()
            if not clean:
                continue
            # Avoid duplicate lines within the same page batch
            line_key = (page_idx, clean)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)

            times, dates = extract_times_and_dates(clean)
            start = times[0] if len(times) >= 1 else None
            end = times[1] if len(times) >= 2 else None

            events.append(
                {
                    "event": clean,
                    "start": start,
                    "end": end,
                    "dates": list(dict.fromkeys(dates)),
                    "ratePercent": None,
                    "behavior": None,
                    "notes": clean,
                    "page": page_idx,
                    "line": line_idx,
                    "confidence": line["confidence"],
                    "bbox": line["bbox"],
                }
            )
            boxes.append(
                {
                    "page": page_idx,
                    "line": line_idx,
                    "text": clean,
                    "bbox": line["bbox"],
                    "confidence": line["confidence"],
                }
            )
    return events, boxes


# ---------------------------------------------------------------------------
# Rendering / preprocessing
# ---------------------------------------------------------------------------

def render_page_to_pil(page: fitz.Page, dpi: int) -> Image.Image:
    """Render a single PyMuPDF page to a grayscale PIL image."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)

    # Safety: downscale if dimensions are huge, to keep OCR cheap
    max_side = 2200
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.BILINEAR)
    return img


def preprocess_image(img: Image.Image) -> Image.Image:
    """Lightweight preprocessing: autocontrast + slight threshold on grayscale image."""
    gray = ImageOps.autocontrast(img)
    # Light threshold to clean background while keeping text
    thresh = gray.point(lambda p: 255 if p > 200 else p)
    return thresh


def compute_nonwhite_ratio(gray: Image.Image) -> float:
    """Estimate how much ink is on the page; used to skip blanks quickly."""
    hist = gray.histogram()
    if not hist:
        return 0.0
    total = sum(hist)
    white = hist[-1] if len(hist) >= 256 else 0
    non_white = total - white
    return non_white / float(total or 1)


def crop_to_content(gray: Image.Image, margin: int = 10) -> Image.Image:
    """
    Crop away large white margins based on non-white pixels.
    Uses only PIL, no numpy.
    """
    # Turn near-white to white, everything else to black
    bw = gray.point(lambda p: 0 if p > 245 else 255, mode="1")
    bbox = bw.getbbox()
    if not bbox:
        return gray
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(gray.width, x2 + margin)
    y2 = min(gray.height, y2 + margin)
    return gray.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def average_confidence(events: List[Dict]) -> float:
    vals = [e.get("confidence") for e in events if e.get("confidence") is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def mark_low_quality_events(events: List[Dict], threshold: float = 0.5):
    for ev in events:
        conf = ev.get("confidence")
        ev["quality"] = "low" if conf is not None and conf < threshold else "ok"


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / float(len(text))


def dedupe_events(events: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for e in events:
        key = (e.get("page"), e.get("line"), e.get("event"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Core PDF OCR pipeline (simple + robust)
# ---------------------------------------------------------------------------

def ocr_pdf_in_batches(
    content: bytes,
    max_pages: int,
    tess_cfg: str,
    time_budget: int,
) -> Tuple[List[Dict], List[Dict], List[str], int]:
    """
    Process PDF pages one by one with a time budget.

    Strategy:
    - Use text layer when present (skip OCR on text-rich pages).
    - Skip clearly blank pages.
    - Single OCR pass per page (no dense retries).
    - Crop to content to avoid OCRing huge white margins.
    """
    start_time = time.monotonic()
    events: List[Dict] = []
    boxes: List[Dict] = []
    warnings: List[str] = []

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        total_pages = doc.page_count
    except Exception:
        return [], [], ["Failed to open PDF for OCR"], 0

    limit = max_pages if max_pages > 0 else total_pages
    limit = min(limit, total_pages)

    for page_idx in range(limit):
        elapsed = time.monotonic() - start_time
        if elapsed > time_budget:
            warnings.append(f"OCR stopped early after reaching time budget ({time_budget}s).")
            break

        page = doc.load_page(page_idx)

        # Fast path: rich text layer available
        try:
            text_preview = page.get_text("text") or ""
        except Exception:
            text_preview = ""
        if len(text_preview) >= MIN_TEXT_CHARS:
            text_events = extract_text_layer_events(page)
            events.extend(text_events)
            continue

        # Render & preprocess
        img = render_page_to_pil(page, dpi=BASE_DPI)
        gray = preprocess_image(img)

        # Skip pages that are essentially blank
        if compute_nonwhite_ratio(gray) < 0.001:
            warnings.append(f"Page {page_idx + 1} skipped (looks blank).")
            continue

        # Crop to content region to avoid huge white borders
        cropped = crop_to_content(gray)

        # Single OCR pass on cropped region
        ev_batch, box_batch = ocr_images(
            [cropped],
            config=tess_cfg,
            base_page_index=page_idx + 1,
        )
        conf_avg = average_confidence(ev_batch)
        mark_low_quality_events(ev_batch, threshold=0.5)

        # Attach page index metadata (override any local indexing)
        for ev in ev_batch:
            ev["page"] = page_idx + 1
        for b in box_batch:
            b["page"] = page_idx + 1

        # Filter out obvious noise-only lines (numeric blobs without time/date)
        filtered_events = []
        for ev in ev_batch:
            has_time = ev.get("start") or ev.get("end")
            has_date = bool(ev.get("dates"))
            if not has_time and not has_date and _digit_ratio(ev.get("event", "")) > 0.7 and ev.get("confidence", 0) < 0.55:
                continue
            filtered_events.append(ev)
        ev_batch = filtered_events

        events.extend(ev_batch)
        boxes.extend(box_batch)

    events = dedupe_events(events)
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

    is_pdf = file.filename.lower().endswith(".pdf")

    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large for OCR (>{MAX_FILE_BYTES // (1024*1024)} MB). Please trim pages or reduce size.",
        )

    if is_pdf:
        events, boxes, ocr_warnings, page_count = ocr_pdf_in_batches(
            content,
            max_pages=MAX_PDF_PAGES if MAX_PDF_PAGES > 0 else 0,
            tess_cfg=BASE_TESS_CONFIG,
            time_budget=MAX_SECONDS,
        )
    else:
        # Image input
        try:
            pil_img = Image.open(io.BytesIO(content)).convert("L")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load file: {e}") from e
        pre = preprocess_image(pil_img)
        cropped = crop_to_content(pre)
        events, boxes = ocr_images([cropped], config=BASE_TESS_CONFIG, base_page_index=1)
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
                "ocrDpi": BASE_DPI,
            },
        }
    )
