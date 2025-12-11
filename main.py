import io
import os
import re
import time
import tempfile
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import pytesseract
import ocrmypdf

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
# Tunables (env)
# ---------------------------------------------------------------------------

MAX_FILE_BYTES = int(os.environ.get("SOF_MAX_FILE_BYTES", str(40 * 1024 * 1024)))  # 40MB guard
MIN_TEXT_CHARS = int(os.environ.get("SOF_MIN_TEXT_CHARS", "80"))  # text-rich threshold
MAX_SECONDS = int(os.environ.get("SOF_MAX_SECONDS", "300"))  # global soft guard

# Tesseract config for images
BASE_TESS_CONFIG = os.environ.get(
    "SOF_TESS_CONFIG",
    "--oem 1 --psm 6 -c preserve_interword_spaces=1"
)
TESSERACT_TIMEOUT = int(os.environ.get("SOF_TESS_TIMEOUT", "0"))  # 0 = no per-call timeout

# OCRmyPDF options
OCR_LANG = os.environ.get("SOF_OCR_LANG", "eng")
OCR_JOBS = int(os.environ.get("SOF_OCR_JOBS", "2"))


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
# Text-layer extraction (from OCRmyPDF output)
# ---------------------------------------------------------------------------

def extract_text_layer_events_and_boxes(page: fitz.Page) -> Tuple[List[Dict], List[Dict]]:
    """
    Read a page's text layer (rawdict) and turn lines into events + boxes.
    Assumes the PDF has already been OCR'd (e.g., by OCRmyPDF).
    """
    events: List[Dict] = []
    boxes: List[Dict] = []

    try:
        raw = page.get_text("rawdict") or {}
    except Exception:
        return events, boxes

    line_counter = 0
    blocks = raw.get("blocks", [])
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
            ev = {
                "event": text,
                "notes": text,
                "start": start,
                "end": end,
                "dates": list(dict.fromkeys(dates)),
                "page": page.number + 1,
                "line": line_counter,
                "confidence": 0.99,  # OCRmyPDF/Tesseract doesn't give per-token conf here
                "bbox": bbox,
            }
            events.append(ev)
            boxes.append(
                {
                    "page": ev["page"],
                    "line": ev["line"],
                    "text": text,
                    "bbox": bbox,
                    "confidence": ev["confidence"],
                }
            )

    return events, boxes


# ---------------------------------------------------------------------------
# Image OCR helpers (for non-PDF uploads)
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


def preprocess_image(img: Image.Image) -> Image.Image:
    """Lightweight preprocessing: autocontrast for images."""
    if img.mode != "L":
        img = img.convert("L")
    gray = ImageOps.autocontrast(img)
    return gray


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
# Core: OCR PDF with OCRmyPDF, then read with PyMuPDF
# ---------------------------------------------------------------------------

def ocr_pdf_with_ocrmypdf(content: bytes) -> Tuple[List[Dict], List[Dict], List[str], int]:
    """
    Full pipeline for PDFs:

    1) Save uploaded bytes to a temp file.
    2) Run OCRmyPDF to produce an OCR'd PDF with text layer.
    3) Use PyMuPDF to extract text + positions from the OCR'd PDF.
    """
    warnings: List[str] = []
    events: List[Dict] = []
    boxes: List[Dict] = []

    start = time.monotonic()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as in_tmp:
        in_tmp.write(content)
        in_path = in_tmp.name

    out_fd, out_path = tempfile.mkstemp(suffix="-ocr.pdf")
    os.close(out_fd)

    try:
        # Run OCRmyPDF: force OCR on image pages, skip pages that already have text,
        # deskew and rotate, no heavy optimization to keep speed reasonable.
        ocrmypdf.ocr(
            in_path,
            out_path,
            language=OCR_LANG,
            force_ocr=True,
            skip_text=True,
            rotate_pages=True,
            deskew=True,
            optimize=0,
            jobs=OCR_JOBS,
            progress_bar=False,
        )

        # Open the OCR'd PDF and extract text layer
        doc = fitz.open(out_path)
        page_count = doc.page_count

        for page_idx in range(page_count):
            page = doc.load_page(page_idx)
            ev_page, box_page = extract_text_layer_events_and_boxes(page)
            events.extend(ev_page)
            boxes.extend(box_page)

    except Exception as e:
        warnings.append(f"OCRmyPDF failed: {e}")
        page_count = 0
    finally:
        # Clean up temp files
        try:
            os.remove(in_path)
        except Exception:
            pass
        try:
            os.remove(out_path)
        except Exception:
            pass

    # Basic filtering: drop pure numeric noise with no time/date and very low confidence
    filtered_events: List[Dict] = []
    for ev in events:
        has_time = ev.get("start") or ev.get("end")
        has_date = bool(ev.get("dates"))
        if not has_time and not has_date and _digit_ratio(ev.get("event", "")) > 0.7 and ev.get("confidence", 0) < 0.4:
            continue
        filtered_events.append(ev)
    events = dedupe_events(filtered_events)

    elapsed = time.monotonic() - start
    if elapsed > MAX_SECONDS:
        warnings.append(f"Warning: OCR pipeline took {elapsed:.1f}s which exceeds configured MAX_SECONDS={MAX_SECONDS}s.")

    return events, boxes, warnings, page_count


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
        # Main path: OCRmyPDF + PyMuPDF text extraction
        events, boxes, ocr_warnings, page_count = ocr_pdf_with_ocrmypdf(content)
    else:
        # Fallback for image files
        try:
            pil_img = Image.open(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {e}") from e
        pre = preprocess_image(pil_img)
        events, boxes = ocr_images([pre], config=BASE_TESS_CONFIG, base_page_index=1)
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
            },
        }
    )
