import io
import os
import re
from typing import Dict, List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import time

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
DPI = int(os.environ.get("SOF_OCR_DPI", "200"))  # lower DPI for speed unless overridden
MAX_FILE_BYTES = int(os.environ.get("SOF_MAX_FILE_BYTES", str(40 * 1024 * 1024)))  # 40MB guard
MAX_SECONDS = int(os.environ.get("SOF_MAX_SECONDS", "280"))  # keep under 5 min proxy timeout
BATCH_PAGES = int(os.environ.get("SOF_BATCH_PAGES", "3"))  # pages per OCR batch
BASE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG", "--oem 1 --psm 6 -c preserve_interword_spaces=1")
DENSE_TESS_CONFIG = os.environ.get("SOF_TESS_CONFIG_DENSE", "--oem 1 --psm 4 -c preserve_interword_spaces=1")


def extract_pdf_text_events(content: bytes) -> Tuple[List[Dict], int]:
    """Fast path: try native text extraction before OCR for digital PDFs."""
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception:
        return [], 0

    events: List[Dict] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line_idx, line in enumerate(lines, start=1):
            events.append(
                {
                    "event": line,
                    "notes": line,
                    "page": page_idx,
                    "line": line_idx,
                    "confidence": 1.0,  # text extraction is reliable for digital PDFs
                    "bbox": None,
                }
            )
    return events, len(reader.pages)


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


def ocr_pdf_in_batches(content: bytes, dpi: int, max_pages: int, batch_pages: int, tess_cfg: str, dense_cfg: str, time_budget: int):
    """Process PDF pages in small batches to avoid long single calls and allow early exit."""
    start_time = time.monotonic()
    events: List[Dict] = []
    boxes: List[Dict] = []
    warnings: List[str] = []

    try:
        reader = PdfReader(io.BytesIO(content))
        total_pages = len(reader.pages)
    except Exception:
        total_pages = None

    page_num = 1
    while True:
        if max_pages > 0 and page_num > max_pages:
            break
        if time.monotonic() - start_time > time_budget:
            warnings.append(f"OCR stopped early after reaching time budget ({time_budget}s).")
            break

        # Build a batch of pages
        last_page = page_num + batch_pages - 1
        if max_pages > 0:
            last_page = min(last_page, max_pages)
        try:
            imgs = convert_from_bytes(
                content,
                fmt="png",
                dpi=dpi,
                first_page=page_num,
                last_page=last_page,
            )
        except Exception:
            break

        # Optional: auto-rotate via OSD if needed
        def maybe_upright(img: Image.Image) -> Image.Image:
            try:
                osd = pytesseract.image_to_osd(img, config="--psm 0")
                m = re.search(r"Rotate: (\d+)", osd)
                if m:
                    rot = int(m.group(1)) % 360
                    if rot:
                        return img.rotate(-rot, expand=True)
            except Exception:
                pass
            return img

        imgs = [maybe_upright(im) for im in imgs]

        ev_batch, box_batch = ocr_images(imgs, config=tess_cfg)
        events.extend(ev_batch)
        boxes.extend(box_batch)

        page_num = last_page + 1
        if total_pages and page_num > total_pages:
            break

    # If very little came back, run a dense pass across the same pages (respecting time budget)
    if len(events) < 5 and (time.monotonic() - start_time) < time_budget:
        page_num = 1
        events = []
        boxes = []
        while True:
            if max_pages > 0 and page_num > max_pages:
                break
            if time.monotonic() - start_time > time_budget:
                warnings.append(f"OCR stopped early after reaching time budget ({time_budget}s).")
                break
            last_page = page_num + batch_pages - 1
            if max_pages > 0:
                last_page = min(last_page, max_pages)
            try:
                imgs = convert_from_bytes(
                    content,
                    fmt="png",
                    dpi=max(dpi, 300),
                    first_page=page_num,
                    last_page=last_page,
                )
            except Exception:
                break
            imgs = [maybe_upright(im) for im in imgs]
            ev_batch, box_batch = ocr_images(imgs, config=dense_cfg)
            events.extend(ev_batch)
            boxes.extend(box_batch)
            page_num = last_page + 1
            if total_pages and page_num > total_pages:
                break

    return events, boxes, warnings


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    is_pdf = file.filename.lower().endswith(".pdf")
    text_events_for_merge: List[Dict] = []
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

    try:
        if is_pdf:
            # Convert pages (all by default; cap if env explicitly set)
            images = convert_from_bytes(
                content,
                fmt="png",
                dpi=DPI,
                first_page=1,
                last_page=MAX_PDF_PAGES if MAX_PDF_PAGES > 0 else None,
            )
        else:
            images = [Image.open(io.BytesIO(content))]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {e}") from e

    events, boxes, ocr_warnings = ocr_pdf_in_batches(
        content,
        dpi=DPI,
        max_pages=MAX_PDF_PAGES if MAX_PDF_PAGES > 0 else 0,
        batch_pages=max(BATCH_PAGES, 1),
        tess_cfg=BASE_TESS_CONFIG,
        dense_cfg=DENSE_TESS_CONFIG,
        time_budget=MAX_SECONDS,
    )

    # Merge any sparse text-layer events we collected earlier
    if text_events_for_merge:
        events = text_events_for_merge + events

    return JSONResponse(
        {
            "events": events,
            "boxes": boxes,
            "warnings": ocr_warnings,
            "meta": {"sourcePages": None, "durationMs": None},
        }
    )
