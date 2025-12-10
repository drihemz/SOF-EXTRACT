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


def ocr_images(images: List[Image.Image]):
    events = []
    boxes = []
    for page_idx, img in enumerate(images, start=1):
        # Better OCR defaults: OEM 1 (LSTM), PSM 6 (block of text)
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            config="--oem 1 --psm 6",
            lang="eng",
        )
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


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    is_pdf = file.filename.lower().endswith(".pdf")
    if is_pdf:
        text_events, page_count = extract_pdf_text_events(content)
        if len(text_events) > 0:
            return JSONResponse(
                {
                    "events": text_events,
                    "boxes": [],
                    "warnings": [],
                    "meta": {"sourcePages": page_count or None, "durationMs": None},
                }
            )

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

    events, boxes = ocr_images(images)
    if len(events) == 0:
        # Fallback: plain text extraction per page to avoid empty responses
        for page_idx, img in enumerate(images, start=1):
            try:
                text = pytesseract.image_to_string(img, config="--oem 1 --psm 6", lang="eng")
            except Exception:
                text = ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for line_idx, line_text in enumerate(lines, start=1):
                events.append(
                    {
                        "event": line_text,
                        "notes": line_text,
                        "page": page_idx,
                        "line": line_idx,
                        "confidence": None,
                        "bbox": None,
                    }
                )

    return JSONResponse(
        {
            "events": events,
            "boxes": boxes,
            "warnings": [],
            "meta": {"sourcePages": len(images), "durationMs": None},
        }
    )
