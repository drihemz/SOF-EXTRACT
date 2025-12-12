import io
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import boto3
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image, ImageOps

app = FastAPI()

# CORS: tighten origins to your frontend domains if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/")
async def health():
    return {"status": "ok"}

TIME_REGEX = re.compile(r"(\d{1,2}[:h\.]\d{2})", re.IGNORECASE)
DATE_REGEX = re.compile(r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})")
HEADER_NOISE_WORDS = {
    "port",
    "of",
    "santos",
    "event",
    "events",
    "time",
    "date",
    "dates",
    "flag",
    "hold",
    "wind",
    "winds",
    "bill",
    "lading",
}

# Tunables via env for performance safeguards
# Set SOF_MAX_PDF_PAGES=0 (default) to process all pages. Set a positive number to cap for performance if desired.
MAX_PDF_PAGES = int(os.environ.get("SOF_MAX_PDF_PAGES", "0"))
PREVIEW_DPI = int(os.environ.get("SOF_PREVIEW_DPI", "96"))  # low-DPI preview for cheap blank detection
BASE_DPI = int(os.environ.get("SOF_OCR_DPI", "150"))  # default raster DPI for OCR (conservative, smaller)
DENSE_DPI = int(os.environ.get("SOF_OCR_DENSE_DPI", "240"))  # kept for compatibility; dense pass disabled in fast path
MAX_FILE_BYTES = int(os.environ.get("SOF_MAX_FILE_BYTES", str(40 * 1024 * 1024)))  # 40MB guard
MAX_SECONDS = int(os.environ.get("SOF_MAX_SECONDS", "260"))  # keep under proxy timeout
PER_PAGE_SECONDS = int(os.environ.get("SOF_PAGE_SECONDS", "40"))  # hard ceiling per page to avoid tail spikes
MIN_TEXT_CHARS = int(os.environ.get("SOF_MIN_TEXT_CHARS", "120"))  # treat page as text-rich above this
PADDLE_LANG = os.environ.get("SOF_PADDLE_LANG", "en")
PADDLE_REC_THREADS = int(os.environ.get("SOF_PADDLE_THREADS", "4"))
# Legacy placeholders for backward compatibility (not used with PaddleOCR)
BASE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG", "")
DENSE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG_DENSE", "")
USE_TEXTRACT = os.environ.get("USE_TEXTRACT", "false").lower() in {"1", "true", "yes"}
TEXTRACT_REGION = os.environ.get("TEXTRACT_REGION", os.environ.get("AWS_REGION", "eu-north-1"))
TEXTRACT_MAX_SYNC_BYTES = 5 * 1024 * 1024  # Textract sync byte limit


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


def _normalize_times(times: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Normalize time strings like 15h20 or 00.00 to HH:MM and drop impossible times."""
    start = end = None
    if len(times) >= 1:
        start = _sanitize_time(times[0].lower().replace("h", ":").replace(".", ":"))
    if len(times) >= 2:
        end = _sanitize_time(times[1].lower().replace("h", ":").replace(".", ":"))
    return start, end


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


def extract_text_layer_events(page: fitz.Page) -> List[Dict]:
    """Use the PDF text layer when present to avoid OCR cost and noise."""
    events: List[Dict] = []
    raw = {}
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


def ocr_images(images: List[Image.Image], config: str = "--oem 1 --psm 6", base_page_index: int = 1):
    events = []
    boxes = []
    seen_lines = set()
    for offset, img in enumerate(images):
        page_idx = base_page_index + offset
        np_img = np.array(img)
        try:
            ocr_result = OCR_ENGINE.ocr(np_img, cls=True)
        except Exception as e:
            boxes.append(
                {
                    "page": page_idx,
                    "line": 0,
                    "text": f"OCR failed on page {page_idx}: {e}",
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "confidence": 0.0,
                }
            )
            continue

        # Paddle returns a list per image; each entry is list of [bbox, (text, conf)]
        lines = ocr_result[0] if ocr_result else []
        for line_idx, entry in enumerate(lines, start=1):
            if not entry or len(entry) < 2:
                continue
            bbox_pts, text_conf = entry
            if not bbox_pts or not text_conf:
                continue
            clean = (text_conf[0] or "").strip()
            conf = float(text_conf[1]) if text_conf[1] is not None else None
            if not clean:
                continue

            line_key = (page_idx, clean)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)

            times, dates = extract_times_and_dates(clean)
            start = times[0] if len(times) >= 1 else None
            end = times[1] if len(times) >= 2 else None

            xs = [pt[0] for pt in bbox_pts]
            ys = [pt[1] for pt in bbox_pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            bbox = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

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
                    "confidence": conf,
                    "bbox": bbox,
                }
            )
            boxes.append(
                {
                    "page": page_idx,
                    "line": line_idx,
                    "text": clean,
                    "bbox": bbox,
                    "confidence": conf,
                }
            )
    return events, boxes


def render_page_to_pil(page: fitz.Page, dpi: int) -> Image.Image:
    """Render a single PyMuPDF page to a grayscale PIL image, downscaled if huge."""
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)

    # Safety: downscale if dimensions are excessively large to keep OCR cheap
    max_side = 1800
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.BILINEAR)
    return img


def render_preview_gray(page: fitz.Page, dpi: int) -> Image.Image:
    """Low-DPI grayscale render for cheap blank detection."""
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


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


def _find_invalid_time_tokens(text: str) -> List[str]:
    """Return time-like tokens that fail HH:MM validation (e.g., 78:06, 59:68)."""
    bad: List[str] = []
    for raw in TIME_REGEX.findall(text):
        normalized = raw.lower().replace("h", ":").replace(".", ":")
        parts = normalized.split(":")
        if len(parts) != 2:
            continue
        try:
            h = int(parts[0])
            m = int(parts[1])
        except ValueError:
            continue
        if not (0 <= h <= 23 and 0 <= m <= 59):
            bad.append(raw)
    return bad


def _filter_events(ev_batch: List[Dict]) -> List[Dict]:
    """Drop obvious headers/noise and rows with impossible times; annotate warnings."""
    filtered: List[Dict] = []
    for ev in ev_batch:
        text = (ev.get("event") or "").strip()
        tokens = re.findall(r"[A-Za-z']+", text)
        lower_tokens = [t.lower() for t in tokens]
        has_time = bool(ev.get("start") or ev.get("end"))
        has_date = bool(ev.get("dates"))
        conf = ev.get("confidence") or 0.0

        invalid_tokens = _find_invalid_time_tokens(text)
        if invalid_tokens:
            ev.setdefault("warnings", []).append(f"Invalid time(s): {', '.join(invalid_tokens)}")
            # If the only time-like strings are invalid, drop the row entirely.
            if not has_time and not has_date:
                continue

        # Require a valid time to keep the row (date-only or header-only lines are dropped).
        if not has_time:
            continue

        # Hard gate: if no time/date, drop short/noisy headers and low-confidence single/dual tokens.
        if not has_time and not has_date:
            if len(tokens) <= 2:
                if all(t in HEADER_NOISE_WORDS for t in lower_tokens):
                    continue
                if conf < 0.85:
                    continue
            if not tokens:
                continue
        filtered.append(ev)
    return filtered


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


# PaddleOCR engine (lazy init)
OCR_ENGINE = PaddleOCR(
    use_angle_cls=True,
    lang=PADDLE_LANG,
)

_textract_client = None


def get_textract_client():
    global _textract_client
    if _textract_client is None:
        _textract_client = boto3.client("textract", region_name=TEXTRACT_REGION)
    return _textract_client


def textract_ocr(content: bytes):
    """Use Amazon Textract detect_document_text to extract lines with bbox/conf/page."""
    warnings: List[str] = []
    if len(content) > TEXTRACT_MAX_SYNC_BYTES:
        raise HTTPException(status_code=413, detail="File too large for Textract sync (5MB limit).")
    client = get_textract_client()
    try:
        resp = client.detect_document_text(Document={"Bytes": content})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Textract error: {e}") from e

    blocks = resp.get("Blocks", [])
    events: List[Dict] = []
    boxes: List[Dict] = []
    line_counters: Dict[int, int] = {}
    for b in blocks:
        if b.get("BlockType") != "LINE":
            continue
        text = (b.get("Text") or "").strip()
        if not text:
            continue
        page = b.get("Page", 1)
        line_counters[page] = line_counters.get(page, 0) + 1
        line_no = line_counters[page]
        conf = float(b.get("Confidence", 0.0)) / 100.0
        geom = b.get("Geometry", {}).get("BoundingBox", {}) or {}
        # Textract bbox is relative 0-1; keep as-is but scale to 1000 for numeric width/height
        x = geom.get("Left", 0) * 1000
        y = geom.get("Top", 0) * 1000
        w = geom.get("Width", 0) * 1000
        h = geom.get("Height", 0) * 1000
        bbox = {"x": x, "y": y, "width": w, "height": h}

        times, dates = extract_times_and_dates(text)
        start = times[0] if len(times) >= 1 else None
        end = times[1] if len(times) >= 2 else None

        events.append(
            {
                "event": text,
                "start": start,
                "end": end,
                "dates": list(dict.fromkeys(dates)),
                "ratePercent": None,
                "behavior": None,
                "notes": text,
                "page": page,
                "line": line_no,
                "confidence": conf,
                "bbox": bbox,
            }
        )
        boxes.append(
            {
                "page": page,
                "line": line_no,
                "text": text,
                "bbox": bbox,
                "confidence": conf,
            }
        )

    page_count = resp.get("DocumentMetadata", {}).get("Pages", 0) or max(line_counters.keys() or [0])
    return events, boxes, warnings, page_count


def ocr_pdf_in_batches(content: bytes, max_pages: int, tess_cfg: str, dense_cfg: str, time_budget: int):
    """Process PDF pages one by one with a time budget using PaddleOCR; keeps all pages unless budget is hit."""
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

        page_start = time.monotonic()
        page = doc.load_page(page_idx)

        # Fast path: rich text layer available
        text_preview = ""
        try:
            text_preview = page.get_text("text") or ""
        except Exception:
            text_preview = ""
        if len(text_preview) >= MIN_TEXT_CHARS:
            text_events = extract_text_layer_events(page)
            events.extend(text_events)
            # Text-layer path does not produce boxes; acceptable since confidence is high.
            continue

        # Cheap preview to skip blanks
        preview_gray = render_preview_gray(page, dpi=PREVIEW_DPI)
        if compute_nonwhite_ratio(preview_gray) < 0.002:
            warnings.append(f"Page {page_idx + 1} skipped (looks blank on preview).")
            continue

        # Single render at base DPI
        img = render_page_to_pil(page, dpi=BASE_DPI)
        gray = preprocess_image(img)

        # Single OCR pass, bounded by Tesseract timeout
        ev_batch, box_batch = ocr_images(
            [gray],
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
            has_date = ev.get("dates")
            if not has_time and not has_date and _digit_ratio(ev.get("event", "")) > 0.7 and ev.get("confidence", 0) < 0.55:
                continue
            filtered_events.append(ev)
        ev_batch = filtered_events
        ev_batch = _filter_events(ev_batch)

        events.extend(ev_batch)
        boxes.extend(box_batch)

        if time.monotonic() - page_start > PER_PAGE_SECONDS:
            warnings.append(
                f"Page {page_idx + 1} processing reached per-page time budget ({PER_PAGE_SECONDS}s); OCR result kept, no extra work attempted."
            )

    events = _filter_events(dedupe_events(events))
    keep_keys = {(e.get("page"), e.get("line"), e.get("event")) for e in events}
    boxes = [b for b in boxes if (b.get("page"), b.get("line"), b.get("text")) in keep_keys]
    return events, boxes, warnings, total_pages


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

    # Branch: PDFs use batch OCR; images use single-pass OCR
    if is_pdf:
        if USE_TEXTRACT:
            events, boxes, ocr_warnings, page_count = textract_ocr(content)
        else:
            events, boxes, ocr_warnings, page_count = ocr_pdf_in_batches(
                content,
                max_pages=MAX_PDF_PAGES if MAX_PDF_PAGES > 0 else 0,
                tess_cfg=BASE_TESS_CONFIG,
                dense_cfg=DENSE_TESS_CONFIG,
            time_budget=MAX_SECONDS,
        )
    else:
        try:
            images = [Image.open(io.BytesIO(content))]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load file: {e}") from e
        events, boxes = ocr_images(images)
        ocr_warnings = []
        page_count = len(images)

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
                "perPageSeconds": PER_PAGE_SECONDS,
            },
        }
    )
