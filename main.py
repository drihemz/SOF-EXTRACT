import io
import os
import re
import time
from typing import Dict, List, Tuple

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

TIME_REGEX = re.compile(r"(\d{1,2}[:\.]\d{2})")
DATE_REGEX = re.compile(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})")

# Tunables via env for performance safeguards
# Set SOF_MAX_PDF_PAGES=0 (default) to process all pages. Set a positive number to cap for performance if desired.
MAX_PDF_PAGES = int(os.environ.get("SOF_MAX_PDF_PAGES", "0"))
BASE_DPI = int(os.environ.get("SOF_OCR_DPI", "220"))  # default raster DPI for OCR
DENSE_DPI = int(os.environ.get("SOF_OCR_DENSE_DPI", "280"))  # slightly higher DPI for dense tables
MAX_FILE_BYTES = int(os.environ.get("SOF_MAX_FILE_BYTES", str(40 * 1024 * 1024)))  # 40MB guard
MAX_SECONDS = int(os.environ.get("SOF_MAX_SECONDS", "270"))  # keep under proxy timeout
PER_PAGE_SECONDS = int(os.environ.get("SOF_PAGE_SECONDS", "35"))  # hard ceiling per page to avoid tail spikes
MIN_TEXT_CHARS = int(os.environ.get("SOF_MIN_TEXT_CHARS", "120"))  # treat page as text-rich above this
BASE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG", "--oem 1 --psm 6 -c preserve_interword_spaces=1")
DENSE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG_DENSE", "--oem 1 --psm 4 -c preserve_interword_spaces=1")


def extract_text_layer_events(page: fitz.Page) -> List[Dict]:
    """Use the PDF text layer when present to avoid OCR cost and noise."""
    events: List[Dict] = []
    raw = {}
    try:
        raw = page.get_text("rawdict") or {}
    except Exception:
        return events

    blocks = raw.get("blocks", [])
    for block in blocks:
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            x1, y1, x2, y2 = line.get("bbox", (0, 0, 0, 0))
            bbox = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
            events.append(
                {
                    "event": text,
                    "notes": text,
                    "page": page.number + 1,
                    "line": len(events) + 1,
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


def ocr_images(images: List[Image.Image], config: str = "--oem 1 --psm 6"):
    events = []
    boxes = []
    for page_idx, img in enumerate(images, start=1):
        # Better OCR defaults: OEM 1 (LSTM), configurable PSM
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config, lang="eng")
        lines = parse_line_groups(data)
        for line_idx, line in enumerate(lines, start=1):
            clean = line["text"].strip()
            if not clean:
                continue
            times = TIME_REGEX.findall(clean)
            dates = DATE_REGEX.findall(clean)
            start = end = None
            if len(times) >= 1:
                start = times[0].replace(".", ":")
            if len(times) >= 2:
                end = times[1].replace(".", ":")
            events.append(
                {
                    "event": clean,
                    "start": start,
                    "end": end,
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
        return [], [], ["Failed to open PDF for OCR"]

    limit = max_pages if max_pages > 0 else total_pages
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

        # Render page at base DPI
        img = render_page_to_pil(page, dpi=BASE_DPI)
        gray = preprocess_image(img)

        # Skip blank or near-blank pages quickly
        if compute_nonwhite_ratio(gray) < 0.005:
            warnings.append(f"Page {page_idx + 1} skipped (looks blank).")
            continue

        # Dense table heuristic: lots of edges imply smaller text → bump DPI for this page
        if edge_density(gray) > 0.08 and DENSE_DPI > BASE_DPI:
            img = render_page_to_pil(page, dpi=DENSE_DPI)
            gray = preprocess_image(img)

        # Primary OCR
        ev_batch, box_batch = ocr_images([gray], config=tess_cfg)
        conf_avg = average_confidence(ev_batch)

        # Targeted re-run with denser PSM/DPI when the pass looks weak and we still have budget
        if (len(ev_batch) < 3 or conf_avg < 0.55) and (time.monotonic() - page_start) < PER_PAGE_SECONDS:
            if DENSE_DPI > BASE_DPI:
                img = render_page_to_pil(page, dpi=DENSE_DPI)
                gray = preprocess_image(img)
            ev_retry, box_retry = ocr_images([gray], config=dense_cfg)
            if len(ev_retry) > len(ev_batch) or average_confidence(ev_retry) > conf_avg:
                ev_batch, box_batch = ev_retry, box_retry

        # Attach page index metadata
        for ev in ev_batch:
            ev.setdefault("page", page_idx + 1)
        for b in box_batch:
            b.setdefault("page", page_idx + 1)

        events.extend(ev_batch)
        boxes.extend(box_batch)

    return events, boxes, warnings, total_pages


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
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

    return JSONResponse(
        {
            "events": events,
            "boxes": boxes,
            "warnings": ocr_warnings,
            "meta": {"sourcePages": page_count, "durationMs": None},
        }
    )
