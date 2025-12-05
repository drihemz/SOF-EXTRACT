import io
import re
import tempfile
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

app = FastAPI()

TIME_REGEX = re.compile(r"(\d{1,2}[:\.]\d{2})")
DATE_REGEX = re.compile(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})")

def ocr_images(images: List[Image.Image]):
    events = []
    for page_idx, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img)
        for line_idx, line in enumerate(text.splitlines(), start=1):
            clean = line.strip()
            if not clean:
                continue
            times = TIME_REGEX.findall(clean)
            dates = DATE_REGEX.findall(clean)
            start = end = None
            if len(times) >= 1:
                start = times[0].replace(".", ":")
            if len(times) >= 2:
                end = times[1].replace(".", ":")
            event_name = clean
            events.append({
                "event": event_name,
                "start": start,
                "end": end,
                "ratePercent": None,
                "behavior": None,
                "notes": clean,
                "page": page_idx,
                "line": line_idx,
                "confidence": None,
            })
    return events

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(content, fmt="png")
        else:
            images = [Image.open(io.BytesIO(content))]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {e}") from e

    events = ocr_images(images)
    return JSONResponse({
        "events": events,
        "warnings": [],
        "meta": {"sourcePages": len(images), "durationMs": None}
    })
