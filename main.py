import io
import os
import re
import time
from typing import Dict, List, Tuple

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

TIME_REGEX = re.compile(r"(\d{1,2}[:\.]\d{2})")
DATE_REGEX = re.compile(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})")

# Tunables via env for performance safeguards
# Set SOF_MAX_PDF_PAGES=0 (default) to process all pages. Set a positive number to cap for performance if desired.
MAX_PDF_PAGES = int(os.environ.get("SOF_MAX_PDF_PAGES", "0"))
DPI = int(os.environ.get("SOF_OCR_DPI", "200"))  # default raster DPI for OCR
MAX_FILE_BYTES = int(os.environ.get("SOF_MAX_FILE_BYTES", str(40 * 1024 * 1024)))  # 40MB guard
MAX_SECONDS = int(os.environ.get("SOF_MAX_SECONDS", "270"))  # keep under proxy timeout
BASE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG", "--oem 1 --psm 6 -c preserve_interword_spaces=1")
DENSE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG_DENSE", "--oem 1 --psm 4 -c preserve_interword_spaces=1")


def extract_pdf_text_events(content: bytes) -> Tuple[List[Dict], int]:
    """Fast path: try native text extraction before OCR for digital PDFs."""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception:
        return [], 0

    events: List[Dict] = []
    for page_idx in range(doc.page_count):
        try:
            page = doc.load_page(page_idx)
            text = page.get_text("text") or ""
        except Exception:
            text = ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line_idx, line in enumerate(lines, start=1):
            events.append(
                {
                    "event": line,
                    "notes": line,
                    "page": page_idx + 1,
                    "line": line_idx,
                    "confidence": 1.0,
                    "bbox": None,
                }
            )
    return events, doc.page_count


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


def ocr_pdf_in_batches(content: bytes, dpi: int, max_pages: int, tess_cfg: str, dense_cfg: str, time_budget: int):
    """Process PDF pages one by one with a time budget; keeps all pages unless budget is hit."""
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
        if time.monotonic() - start_time > time_budget:
            warnings.append(f"OCR stopped early after reaching time budget ({time_budget}s).")
            break

        page = doc.load_page(page_idx)
        # Render page
        img = render_page_to_pil(page, dpi=dpi)
        img = preprocess_image(img)

        # Primary OCR
        ev_batch, box_batch = ocr_images([img], config=tess_cfg)
        # If sparse, retry dense
        if len(ev_batch) < 3:
            ev_batch, box_batch = ocr_images([img], config=dense_cfg)

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
    text_events_for_merge: List[Dict] = []
    page_count = None

    if is_pdf:
        text_events, page_count = extract_pdf_text_events(content)
        if len(text_events) >= 5:
            return JSONResponse(
                {
                    "events": text_events,
                    "boxes": [],
                    "warnings": [],
                    "meta": {"sourcePages": page_count or None, "durationMs": None},
                }
            )
        # If text layer is too sparse, keep these lines and fall through to OCR to try to enrich
        text_events_for_merge = text_events or []

    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large for OCR (>{MAX_FILE_BYTES // (1024*1024)} MB). Please trim pages or reduce size.",
        )

    # Branch: PDFs use batch OCR; images use single-pass OCR
    if is_pdf:
        events, boxes, ocr_warnings, page_count = ocr_pdf_in_batches(
            content,
            dpi=DPI,
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

    # Merge any sparse text-layer events we collected earlier
    if text_events_for_merge:
        events = text_events_for_merge + events

    return JSONResponse(
        {
            "events": events,
            "boxes": boxes,
            "warnings": ocr_warnings,
            "meta": {"sourcePages": page_count, "durationMs": None},
        }
    )
