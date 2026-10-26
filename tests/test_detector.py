import dataclasses
from pathlib import Path

import pytest
from PIL import Image

from detector import analyze, ocr

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_hype_review_is_fake():
    result = analyze("AMAZING product!!! Best purchase ever, highly recommend to everyone. "
                     "Absolutely perfect, five stars!!! You won't regret it!", rating=5)
    assert result.is_fake


def test_spam_review_is_fake():
    assert analyze("Great product, works great. Use my code SAVE20 at www.dealz-shop.com!").is_fake


def test_spam_is_not_averaged_away_even_when_relaxed():
    # A promo code + shop link is damning on its own; the clean checks must not dilute it.
    result = analyze("Great product, works great. Use my code SAVE20 at www.dealz-shop.com for a discount!")
    assert result.fake_probability >= 0.9
    assert dataclasses.replace(result, threshold=0.65).is_fake


def test_isbn_is_not_a_phone_number():
    result = analyze("Great book, ISBN 0307887898, the second half drags a little but worth reading.")
    assert not next(s for s in result.signals if s.key == "promo").counts


@pytest.mark.parametrize("threshold", [0.35, 0.5, 0.65])
@pytest.mark.parametrize("text", [
    "Great product, works great. Use my code SAVE20 at www.dealz-shop.com!",
    "The lid is a bit stiff and the cable is short, but after two months it still boils fast.",
])
def test_confidence_never_below_half(text, threshold):
    assert dataclasses.replace(analyze(text), threshold=threshold).confidence >= 0.5


def test_detailed_balanced_review_is_real():
    result = analyze(
        "I've been using this kettle for about three months. It boils 1.5 liters in roughly "
        "four minutes, which is faster than my old one. The lid is a bit stiff to open and the "
        "cable is short, but for the price it does the job well.")
    assert not result.is_fake


def test_rating_mismatch_fires():
    result = analyze("Broke after two days and support never replied. Terrible.", rating=5)
    assert next(s for s in result.signals if s.key == "rating").triggered


def test_too_short_is_rejected():
    with pytest.raises(ValueError):
        analyze("Nice")


def test_ocr_reads_sample_screenshot():
    result = ocr.extract_review(Image.open(SAMPLES / "sample_fake_review.png"))
    assert "highly recommend" in result.text
    assert "helpful" not in result.text.lower()
    assert result.rating == 5
    assert result.verified_badge
