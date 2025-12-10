import io
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageOps
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

TIME_REGEX = re.compile(r"(\d{1,2}[:h\.]\d{2})", re.IGNORECASE)
DATE_REGEX = re.compile(r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})")

# Tunables via env for performance safeguards
# Set SOF_MAX_PDF_PAGES=0 (default) to process all pages. Set a positive number to cap for performance if desired.
MAX_PDF_PAGES = int(os.environ.get("SOF_MAX_PDF_PAGES", "0"))
BASE_DPI = int(os.environ.get("SOF_OCR_DPI", "200"))  # default raster DPI for OCR (conservative)
DENSE_DPI = int(os.environ.get("SOF_OCR_DENSE_DPI", "240"))  # slightly higher DPI for dense tables
MAX_FILE_BYTES = int(os.environ.get("SOF_MAX_FILE_BYTES", str(40 * 1024 * 1024)))  # 40MB guard
MAX_SECONDS = int(os.environ.get("SOF_MAX_SECONDS", "180"))  # keep under proxy timeout
PER_PAGE_SECONDS = int(os.environ.get("SOF_PAGE_SECONDS", "20"))  # hard ceiling per page to avoid tail spikes
MIN_TEXT_CHARS = int(os.environ.get("SOF_MIN_TEXT_CHARS", "120"))  # treat page as text-rich above this
BASE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG", "--oem 1 --psm 6 -c preserve_interword_spaces=1")
DENSE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG_DENSE", "--oem 1 --psm 4 -c preserve_interword_spaces=1")
TESSERACT_TIMEOUT = int(os.environ.get("SOF_TESS_TIMEOUT", "12"))  # per-call Tesseract timeout in seconds


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


def ocr_images(images: List[Image.Image], config: str = "--oem 1 --psm 6", base_page_index: int = 1):
    events = []
    boxes = []
    seen_lines = set()
    for offset, img in enumerate(images):
        page_idx = base_page_index + offset
        # Better OCR defaults: OEM 1 (LSTM), configurable PSM
        try:
            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config=config,
                lang="eng",
                timeout=TESSERACT_TIMEOUT,
            )
        except RuntimeError as e:
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


def render_page_to_pil(page: fitz.Page, dpi: int) -> Image.Image:
    """Render a single PyMuPDF page to a PIL image."""
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def preprocess_image(img: Image.Image) -> Image.Image:
    """Lightweight preprocessing: grayscale + autocontrast + slight threshold."""
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
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


def edge_density(gray: Image.Image) -> float:
    """Cheap proxy for table/line density to decide if we need higher DPI."""
    edges = gray.filter(ImageFilter.FIND_EDGES)
    hist = edges.histogram()
    total = sum(hist)
    weighted = sum(i * v for i, v in enumerate(hist))
    return (weighted / float(total or 1)) / 255.0


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


def ocr_pdf_in_batches(content: bytes, max_pages: int, tess_cfg: str, dense_cfg: str, time_budget: int):
    """Process PDF pages one by one with a time budget; keeps all pages unless budget is hit.

    Upgrades:
    - Use text layer when present (skip OCR on text-rich pages).
    - Skip near-blank pages.
    - Adaptive DPI/PSM retry when confidence/coverage is low.
    - Hard per-page budget to avoid tail latency.
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
        near_budget = (time_budget - elapsed) < (PER_PAGE_SECONDS * 1.5)

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

        # Render page at base DPI
        used_dpi = BASE_DPI
        img = render_page_to_pil(page, dpi=BASE_DPI)
        gray = preprocess_image(img)

        # Skip blank or near-blank pages quickly
        if compute_nonwhite_ratio(gray) < 0.005:
            warnings.append(f"Page {page_idx + 1} skipped (looks blank).")
            continue

        # Dense table heuristic: lots of edges imply smaller text → bump DPI for this page
        if not near_budget and edge_density(gray) > 0.08 and DENSE_DPI > BASE_DPI:
            img = render_page_to_pil(page, dpi=DENSE_DPI)
            gray = preprocess_image(img)
            used_dpi = DENSE_DPI

        # Primary OCR
        ev_batch, box_batch = ocr_images([gray], config=tess_cfg, base_page_index=page_idx + 1)
        conf_avg = average_confidence(ev_batch)
        mark_low_quality_events(ev_batch, threshold=0.5)

        # Targeted re-run with denser PSM/DPI when the pass looks weak and we still have budget
        bad_coverage = len(ev_batch) < 2
        bad_conf = conf_avg < 0.40
        if (bad_coverage or bad_conf) and (time.monotonic() - page_start) < PER_PAGE_SECONDS and not near_budget:
            if DENSE_DPI > used_dpi:
                img = render_page_to_pil(page, dpi=DENSE_DPI)
                gray = preprocess_image(img)
                used_dpi = DENSE_DPI
            ev_retry, box_retry = ocr_images([gray], config=dense_cfg, base_page_index=page_idx + 1)
            mark_low_quality_events(ev_retry, threshold=0.5)
            if len(ev_retry) > len(ev_batch) or average_confidence(ev_retry) > conf_avg:
                ev_batch, box_batch = ev_retry, box_retry

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

        events.extend(ev_batch)
        boxes.extend(box_batch)

        if time.monotonic() - page_start > PER_PAGE_SECONDS:
            warnings.append(
                f"Page {page_idx + 1} processing reached per-page time budget ({PER_PAGE_SECONDS}s); no further retries attempted."
            )

    events = dedupe_events(events)
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
