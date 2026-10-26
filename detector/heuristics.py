"""Rule-based signals for spotting fake reviews.

Each signal checks one well-known trait of fake, paid, or bot-written reviews and
returns a score from 0 (looks genuine) to 1 (looks fake) with a plain-English
reason. Signals are deliberately simple so every verdict can be explained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
# VADER is tuned for social media; nudge it for product reviews ("customer support" is
# neutral, "broke" and "defective" are complaints).
_vader.lexicon.update({
    "support": 0.0, "broke": -2.0, "broken": -2.0, "defective": -2.5, "faulty": -2.0,
    "refund": -1.0, "returned": -1.0, "stopped": -1.2, "flimsy": -1.8, "cheaply": -1.5,
    "overpriced": -1.8, "leaks": -1.5, "leaked": -1.5,
})

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

HYPE_WORDS = {
    "amazing", "awesome", "perfect", "perfectly", "excellent", "incredible", "fantastic",
    "outstanding", "wonderful", "superb", "flawless", "unbelievable", "phenomenal",
    "magnificent", "spectacular", "best", "greatest", "brilliant", "extraordinary",
    "terrible", "worst", "horrible", "awful", "garbage", "scam", "useless", "pathetic",
    "disgusting", "trash", "fraud",
}

GENERIC_PHRASES = [
    r"highly recommend(ed)?", r"must[- ]buy", r"must[- ]have", r"best \w+ ever",
    r"five stars?", r"5 stars?", r"worth every penny", r"you won'?t regret",
    r"buy (it|this|one) now", r"don'?t waste your money", r"waste of money",
    r"exceeded (all )?my expectations", r"best purchase", r"changed my life",
    r"love (it|this)", r"great product", r"works great", r"very good product",
    r"value for money", r"go for it", r"totally recommend", r"10/10", r"game[- ]changer",
    r"do not buy", r"stay away", r"would recommend (it|this) to (everyone|anyone)",
    r"(simply|just) the best", r"nothing (bad|negative) to say",
]
GENERIC_RE = re.compile(r"\b(" + "|".join(GENERIC_PHRASES) + r")\b", re.IGNORECASE)

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}

ASPECT_WORDS = {
    "battery", "size", "fit", "fits", "price", "delivery", "shipping", "packaging",
    "package", "screen", "sound", "material", "fabric", "color", "colour", "smell",
    "taste", "weight", "setup", "install", "installation", "service", "refund", "return",
    "returned", "warranty", "instructions", "manual", "button", "buttons", "cable",
    "charger", "charging", "stitching", "zipper", "strap", "handle", "lid", "box",
    "seller", "customer", "support", "app", "bluetooth", "wifi", "noise", "speed",
    "texture", "flavor", "flavour", "plastic", "metal", "leather", "wood", "glass",
    "width", "length", "height", "inch", "inches", "cm", "mm", "kg", "lbs", "pounds",
    "ounces", "oz", "ml", "liters", "watts", "volume", "temperature", "durability",
}
TIME_RE = re.compile(
    r"\b(\d+|a|one|two|three|few|couple of)\s+(minute|hour|day|week|month|year)s?\b"
    r"|\b(yesterday|last (week|month|year)|after using|so far)\b",
    re.IGNORECASE,
)
CONTRAST_RE = re.compile(
    r"\b(but|however|although|though|except|only (downside|issue|problem|complaint)"
    r"|wish|cons?|on the other hand|that said|unfortunately)\b",
    re.IGNORECASE,
)
PROMO_RE = re.compile(
    # Phone numbers need a "+" or separators, so ISBNs and model numbers don't match.
    r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b|\+\d[\d\s-]{8,}\d|\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"
    r"|\b(coupon|promo code|discount code|use (my|the) code|use my link|click (here|the link)"
    r"|visit (our|my)|dm me|whatsapp|telegram|follow me|order now|limited offer)\b",
    re.IGNORECASE,
)
INCENTIVE_RE = re.compile(
    r"\b(received (this|the|it)( \w+)? (for free|free|at a discount|at a discounted price)"
    r"|in exchange for (my|an) (honest|unbiased|fair) (review|opinion)"
    r"|provided (to me )?(by|from) the (seller|manufacturer|company|brand)"
    r"|free (sample|product) (for|in return)|sent (me )?(this|it) (for free|to review))\b",
    re.IGNORECASE,
)


@dataclass
class Signal:
    key: str
    name: str
    score: float          # 0 = genuine-looking, 1 = fake-looking
    weight: float
    detail: str
    # Evidence-only signals (spam links, rating mismatch...) say nothing about a
    # review when absent, so they only join the average when they fire.
    evidence_only: bool = False
    spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return self.score >= 0.5

    @property
    def counts(self) -> bool:
        return not self.evidence_only or self.score > 0


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if len(WORD_RE.findall(s)) >= 2]


def _sentence_tones(text: str) -> list[float]:
    # Score sentence by sentence: on a whole review VADER lets a "never" in one
    # sentence negate "terrible" in the next.
    return [tone for _, tone in sentence_breakdown(text)]


def sentence_breakdown(text: str) -> list[tuple[str, float]]:
    """Each sentence with its VADER tone (-1 negative ... +1 positive)."""
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if WORD_RE.search(s)] or [text]
    return [(s, _vader.polarity_scores(s)["compound"]) for s in parts]


def _tone(text: str) -> float:
    tones = _sentence_tones(text)
    return sum(tones) / len(tones)


def hype_words(text: str, words: list[re.Match]) -> Signal:
    hits = [m for m in words if m.group().lower() in HYPE_WORDS]
    density = len(hits) / max(1, len(words))
    score = _clamp(density / 0.08)
    return Signal(
        "hype", "Extreme / superlative words", score, 1.2,
        f"{len(hits)} extreme word(s) ({density:.0%} of the text). "
        "Fake reviews lean on words like 'amazing', 'perfect' or 'worst'.",
        spans=[m.span() for m in hits],
    )


def generic_phrases(text: str, words: list[re.Match]) -> Signal:
    hits = list(GENERIC_RE.finditer(text))
    score = _clamp(0.35 * len(hits))
    return Signal(
        "generic", "Stock / template phrases", score, 1.2,
        f"{len(hits)} stock phrase(s) such as 'highly recommend' or 'worth every penny'. "
        "Paid reviews often reuse the same template lines.",
        spans=[m.span() for m in hits],
    )


def shouting(text: str, words: list[re.Match]) -> Signal:
    exclaims = text.count("!")
    caps = [m for m in words if len(m.group()) >= 3 and m.group().isupper()]
    sentences = max(1, len(_sentences(text)))
    score = _clamp(0.4 * exclaims / sentences + 4 * len(caps) / max(1, len(words))
                   + (0.3 if re.search(r"[!?]{2,}", text) else 0))
    return Signal(
        "shouting", "Excessive punctuation / CAPS", score, 0.8,
        f"{exclaims} exclamation mark(s) and {len(caps)} ALL-CAPS word(s). "
        "Over-the-top emphasis is common in exaggerated reviews.",
        spans=[m.span() for m in caps],
    )


def first_person(text: str, words: list[re.Match]) -> Signal:
    count = sum(1 for m in words if m.group().lower() in FIRST_PERSON)
    ratio = count / max(1, len(words))
    score = _clamp((ratio - 0.07) / 0.08)
    return Signal(
        "first_person", "Self-focused language", score, 0.5,
        f"'I / me / my' make up {ratio:.0%} of words. Research on deceptive reviews "
        "(Ott et al., 2011) found fakes talk about the writer more than the product.",
    )


def lack_of_specifics(text: str, words: list[re.Match]) -> Signal:
    aspects = [m for m in words if m.group().lower() in ASPECT_WORDS]
    numbers = re.findall(r"\b\d+(\.\d+)?\b", text)
    times = TIME_RE.findall(text)
    found = len(aspects) + len(numbers) + len(times)
    expected = max(1.0, len(words) / 25)
    score = _clamp(1 - found / expected)
    return Signal(
        "specifics", "Lack of concrete details", score, 1.2,
        f"{found} concrete detail(s) (product features, numbers, how long it was used). "
        "Genuine buyers usually mention specifics; fakes stay vague.",
    )


def one_sided(text: str, words: list[re.Match]) -> Signal:
    scores = _sentence_tones(text)
    overall = sum(scores) / len(scores)
    has_pos = any(s > 0.3 for s in scores)
    has_neg = any(s < -0.3 for s in scores)
    mixed = (has_pos and has_neg) or bool(CONTRAST_RE.search(text))
    score = abs(overall) * (0.3 if mixed else 1.0)
    tone = "positive" if overall > 0 else "negative"
    detail = (
        "Mentions both pros and cons, which is typical of real reviews."
        if mixed else
        f"Entirely {tone} with no trade-offs (sentiment {overall:+.2f}). "
        "Real experiences are rarely 100% one-sided."
    )
    return Signal("one_sided", "One-sided sentiment", score, 1.0, detail)


def too_short(text: str, words: list[re.Match]) -> Signal:
    n = len(words)
    score = 0.9 if n < 8 else 0.6 if n < 20 else 0.3 if n < 40 else 0.1
    return Signal(
        "length", "Very short review", score, 0.7,
        f"{n} words. Very short reviews give little evidence of real use.",
    )


def repetition(text: str, words: list[re.Match]) -> Signal:
    tokens = [m.group().lower() for m in words][:120]
    if len(tokens) < 15:
        return Signal("repetition", "Repetitive wording", 0.0, 0.8,
                      "Too short to judge repetition.", evidence_only=True)
    ttr = len(set(tokens)) / len(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    repeated = len(bigrams) - len(set(bigrams))
    score = _clamp((0.6 - ttr) / 0.25 + repeated / 10)
    return Signal(
        "repetition", "Repetitive wording", score, 0.8,
        f"Vocabulary variety {ttr:.0%}, {repeated} repeated word pair(s). "
        "Bots and copy-paste reviews repeat themselves.",
    )


def promotional(text: str, words: list[re.Match]) -> Signal:
    hits = list(PROMO_RE.finditer(text))
    score = 0.0 if not hits else 0.75 if len(hits) == 1 else 0.95
    return Signal(
        "promo", "Links / promo codes / contact info", score, 1.5,
        f"{len(hits)} link(s), code(s) or contact detail(s) found. Reviews that advertise are spam."
        if hits else "No links, promo codes or contact details.",
        evidence_only=True, spans=[m.span() for m in hits],
    )


def incentivized(text: str, words: list[re.Match]) -> Signal:
    hits = list(INCENTIVE_RE.finditer(text))
    return Signal(
        "incentive", "Incentivized review disclosure", 0.65 if hits else 0.0, 1.0,
        "Says the product was free or discounted in exchange for a review, so it is not independent."
        if hits else "No sign the reviewer was paid or given the product.",
        evidence_only=True, spans=[m.span() for m in hits],
    )


def rating_mismatch(text: str, rating: int | None) -> Signal:
    if rating is None:
        return Signal("rating", "Star rating vs. text mismatch", 0.0, 1.5,
                      "No star rating given.", evidence_only=True)
    compound = _tone(text)
    mismatch = (rating >= 4 and compound < -0.3) or (rating <= 2 and compound > 0.4)
    return Signal(
        "rating", "Star rating vs. text mismatch", 0.9 if mismatch else 0.0, 1.5,
        f"{rating}-star rating but the text reads {'negative' if compound < 0 else 'positive'} "
        f"(sentiment {compound:+.2f})." if mismatch else
        f"{rating}-star rating is consistent with the tone of the text.",
        evidence_only=True,
    )


TEXT_SIGNALS = [
    hype_words, generic_phrases, shouting, first_person, lack_of_specifics,
    one_sided, too_short, repetition, promotional, incentivized,
]


def run_signals(text: str, rating: int | None = None) -> list[Signal]:
    words = list(WORD_RE.finditer(text))
    signals = [fn(text, words) for fn in TEXT_SIGNALS]
    signals.append(rating_mismatch(text, rating))
    return signals


def combine(signals: list[Signal]) -> float:
    """Weighted average of the signals that carry evidence, floored by the strongest
    evidence-only signal: a promo link or a rating mismatch is damning on its own and
    must not be averaged away by the checks that found nothing."""
    active = [s for s in signals if s.counts]
    total = sum(s.weight for s in active)
    average = sum(s.score * s.weight for s in active) / total if total else 0.0
    strongest = max((s.score for s in signals if s.evidence_only), default=0.0)
    return max(average, strongest)
