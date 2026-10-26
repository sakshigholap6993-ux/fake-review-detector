"""Pull review text out of a screenshot using RapidOCR (ONNX models ship inside the pip
package, so there is nothing to install on the server and nothing downloaded at runtime)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

# Screenshot chrome from shopping sites that is not part of the review itself.
NOISE_RE = re.compile(
    r"^(helpful|report( abuse)?|verified purchase|read more|see more|show more|translate( review)?"
    r"|was this (review )?helpful\??|yes|no|like|reply|share|top reviews?.*|customer reviews?"
    r"|\d+ (people|person) found this helpful|reviewed in .* on .*|\d[\d,]* (ratings?|reviews?))$",
    re.IGNORECASE,
)
NOISE_WORDS = {"helpful", "report", "abuse", "like", "reply", "share", "translate", "yes", "no", "|"}
RATING_RE = re.compile(r"\b([1-5](?:\.\d)?)\s*(?:out of 5|/\s*5)(?:\s*stars?)?", re.IGNORECASE)
VERIFIED_RE = re.compile(r"verified (purchase|buyer)", re.IGNORECASE)
# The recogniser occasionally garbles a line at one scale and reads it perfectly at
# another, with a low confidence score when it fails, so we try a few scales and keep
# the most confident read.
SCALES = (1.0, 1.5, 2.0)
MAX_SIDE = 4000
GOOD_ENOUGH = 0.93


@dataclass
class OcrResult:
    text: str
    rating: int | None
    verified_badge: bool
    lines: list[str]


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={"Global.log_level": "warning"})


def _best_read(image: Image.Image):
    """OCR the image at a few scales and return the output with the highest mean confidence."""
    image = image.convert("RGB")
    best, best_score = None, -1.0
    for scale in SCALES:
        if scale > 1 and max(image.size) * scale > MAX_SIDE:
            break
        scaled = image if scale == 1 else image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        out = _engine()(np.array(scaled))
        if out.boxes is None:
            continue
        # Weight by text length so one confident short word can't outvote a garbled paragraph.
        score = sum(s * len(t) for s, t in zip(out.scores, out.txts)) / max(1, sum(map(len, out.txts)))
        if score > best_score:
            best, best_score = out, score
        if score >= GOOD_ENOUGH and min(out.scores) >= 0.8:
            break
    return best


def _group_lines(boxes, texts) -> list[str]:
    """Merge OCR fragments that sit on the same visual row, then order rows top-to-bottom."""
    items = []
    for box, text in zip(boxes, texts):
        ys = [p[1] for p in box]
        items.append((min(ys), max(ys), min(p[0] for p in box), text))
    items.sort()
    rows: list[list] = []
    for top, bottom, left, text in items:
        center = (top + bottom) / 2
        if rows and abs(center - rows[-1][0]) < (bottom - top) / 2:
            rows[-1][1].append((left, text))
        else:
            rows.append([center, [(left, text)]])
    return [" ".join(t for _, t in sorted(parts)) for _, parts in rows]


def extract_review(image: Image.Image) -> OcrResult:
    out = _best_read(image)
    lines = [] if out is None else _group_lines(out.boxes.tolist(), out.txts)
    full = "\n".join(lines)

    rating_match = RATING_RE.search(full)
    rating = round(float(rating_match.group(1))) if rating_match else None
    review_lines = []
    for line in lines:
        line = RATING_RE.sub("", line).strip()   # "5.0 out of 5 stars Great!" keeps "Great!"
        if line and not NOISE_RE.match(line) and not set(line.lower().split()) <= NOISE_WORDS:
            review_lines.append(line)
    return OcrResult(
        text=" ".join(review_lines).strip(),
        rating=rating,
        verified_badge=bool(VERIFIED_RE.search(full)),
        lines=lines,
    )
