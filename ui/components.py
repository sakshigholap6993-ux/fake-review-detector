"""Rendering helpers for the Streamlit page: styled cards, highlights and charts."""

from __future__ import annotations

import html
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from detector import Analysis
from detector.heuristics import WORD_RE, sentence_breakdown

CSS_PATH = Path(__file__).with_name("style.css")

# Status palette (state, never reused for series) - always paired with an icon + label.
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
# Diverging pair for sentence tone: red (negative) <-> gray midpoint <-> blue (positive).
NEGATIVE, NEUTRAL, POSITIVE = "#e34948", "#a8a7a1", "#2a78d6"

# Criteria whose matched phrases are highlighted in the review, with their legend label.
HIGHLIGHTS = {
    "hype": "Extreme words",
    "generic": "Stock phrases",
    "shouting": "CAPS",
    "promo": "Links / codes",
    "incentive": "Incentive",
}


def load_css() -> None:
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🕵️ Fake Review Detector</h1>
          <p>Paste a product review or drop in a screenshot. We'll tell you if it smells fishy, and why.</p>
          <div class="chips">
            <span class="chip">🤖 ML model · 95% accurate</span>
            <span class="chip">📏 11 explainable checks</span>
            <span class="chip">🖼️ Screenshot OCR</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status(score: float) -> tuple[str, str, str]:
    """(icon, label, color) for a 0-1 criterion score."""
    if score >= 0.5:
        return "🚩", "Red flag", CRITICAL
    if score >= 0.3:
        return "⚠️", "Watch", WARNING
    return "✅", "OK", GOOD


def verdict_card(a: Analysis) -> None:
    kind = "fake" if a.is_fake else "real"
    title = "🚩 Likely FAKE" if a.is_fake else "✅ Looks REAL"
    sub = f"{a.confidence:.0%} confidence"
    if a.low_confidence:
        sub += " · borderline, treat as a hint"
    pct = a.fake_probability * 100
    st.markdown(
        f"""
        <div class="verdict {kind}">
          <div class="row">
            <div>
              <div class="label">Verdict</div>
              <div class="title">{title}</div>
              <div class="sub">{sub}</div>
            </div>
            <div class="hero-num">{pct:.0f}%<small>FAKE SCORE</small></div>
          </div>
          <div class="meter" role="meter" aria-valuenow="{pct:.0f}" aria-valuemin="0" aria-valuemax="100"
               aria-label="Fake score">
            <div class="fill" style="width:{max(pct, 2):.1f}%"></div>
            <div class="tick" style="left:{a.threshold * 100:.0f}%" title="Fake threshold"></div>
          </div>
          <div class="meter-scale"><span>0% genuine</span>
            <span>threshold {a.threshold:.0%}</span><span>100% fake</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_tiles(a: Analysis) -> None:
    words = len(WORD_RE.findall(a.text))
    tones = [t for _, t in sentence_breakdown(a.text)]
    tone = sum(tones) / len(tones)
    flags = sum(1 for s in a.signals if s.counts and s.triggered)
    ml = "n/a" if a.ml_probability is None else f"{a.ml_probability:.0%}"
    tiles = [
        ("ML model", ml, "machine-written text", "#4f46e5"),
        ("Rule checks", f"{a.rule_score:.0%}", "human hype & spam", "#9333ea"),
        ("Red flags", f"{flags}", f"of {sum(1 for s in a.signals if s.counts)} checks", "#ec4899"),
        ("Tone", f"{tone:+.2f}", f"{words} words", "#f97316"),
    ]
    st.markdown(
        '<div class="tiles">' + "".join(
            f'<div class="tile" style="--t-color:{c}"><div class="t-label">{label}</div>'
            f'<div class="t-value">{value}</div><div class="t-note">{note}</div></div>'
            for label, value, note, c in tiles
        ) + "</div>",
        unsafe_allow_html=True,
    )
    if a.ml_probability is not None:
        driver = "ML model" if a.ml_probability >= a.rule_score else "rule checks"
        st.caption(f"The fake score is the higher of the two detectors (here: the {driver}).")


def highlighted_review(a: Analysis) -> None:
    """The review with each triggering phrase marked in its criterion's color."""
    spans = sorted(
        (start, end, s.key, s.name)
        for s in a.signals if s.triggered and s.key in HIGHLIGHTS
        for start, end in s.spans
    )
    out, pos, used = [], 0, []
    for start, end, key, name in spans:
        if start < pos:
            continue  # overlaps a phrase already marked
        out.append(html.escape(a.text[pos:start]))
        out.append(f'<mark class="m-{key}" title="{html.escape(name)}">{html.escape(a.text[start:end])}</mark>')
        pos = end
        if key not in used:
            used.append(key)
    out.append(html.escape(a.text[pos:]))
    body = "".join(out).replace("\n", "<br>")

    legend = "".join(f'<span class="m-{k}"><i></i>{HIGHLIGHTS[k]}</span>' for k in HIGHLIGHTS if k in used)
    st.markdown(f'<div class="review-box">{body}</div>', unsafe_allow_html=True)
    if legend:
        st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
        st.caption("Hover a highlighted phrase to see which check it tripped.")
    else:
        st.caption("Nothing in the wording tripped a phrase-level check.")


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hoverlabel=dict(align="left"),
        font=dict(size=13),
    )
    return fig


def criteria_chart(a: Analysis) -> None:
    rows = sorted((s for s in a.signals if s.counts), key=lambda s: s.score)
    labels = [f"{_status(s.score)[0]} {s.name}" for s in rows]
    colors = [_status(s.score)[2] for s in rows]
    hover = [
        f"<b>{html.escape(s.name)}</b><br>{s.score:.0%} · {_status(s.score)[1]} · weight {s.weight}"
        f"<br><br>{'<br>'.join(_wrap(html.escape(s.detail), 60))}"
        for s in rows
    ]
    fig = go.Figure(go.Bar(
        x=[s.score * 100 for s in rows], y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{s.score:.0%}" for s in rows], textposition="outside", cliponaxis=False,
        hovertext=hover, hoverinfo="text",
    ))
    fig.add_vline(x=50, line=dict(color="rgba(127,127,127,.6)", width=1, dash="dot"),
                  annotation_text="red-flag line", annotation_position="top", annotation_font_size=11)
    fig.update_xaxes(range=[0, 112], ticksuffix="%", gridcolor="rgba(127,127,127,.15)", zeroline=False)
    fig.update_yaxes(showgrid=False)
    fig = _base_layout(fig, 70 + 34 * len(rows))
    fig.update_layout(bargap=0.35, margin=dict(t=28))
    st.plotly_chart(fig, config={"displayModeBar": False},
                    theme="streamlit")
    st.caption("✅ OK (under 30%) · ⚠️ Watch (30-49%) · 🚩 Red flag (50%+). Hover a bar for the reasoning.")


def sentence_chart(a: Analysis) -> None:
    parts = sentence_breakdown(a.text)
    tones = [t for _, t in parts]
    colors = [POSITIVE if t > 0.05 else NEGATIVE if t < -0.05 else NEUTRAL for t in tones]
    labels = [f"S{i}" for i in range(1, len(parts) + 1)]
    hover = [f"<b>{lab}</b> · tone {t:+.2f}<br>{'<br>'.join(_wrap(html.escape(s), 60))}"
             for lab, (s, t) in zip(labels, parts)]
    fig = go.Figure(go.Bar(
        x=tones, y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        # Labels keep neutral (zero-length) sentences visible.
        text=[f"{t:+.2f}" for t in tones], textposition="outside", cliponaxis=False,
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_xaxes(range=[-1.15, 1.15], tickvals=[-1, -0.5, 0, 0.5, 1],
                     ticktext=["−1 negative", "−0.5", "0", "+0.5", "+1 positive"],
                     gridcolor="rgba(127,127,127,.15)", zeroline=True,
                     zerolinecolor="rgba(127,127,127,.6)", zerolinewidth=1)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(bargap=0.35)
    st.plotly_chart(_base_layout(fig, 70 + 32 * len(parts)), config={"displayModeBar": False},
                    theme="streamlit")
    one_sided = next(s for s in a.signals if s.key == "one_sided")
    st.caption("Each bar is one sentence (hover to read it). Blue = positive, red = negative, "
               "gray = neutral. " + one_sided.detail)


def criteria_list(a: Analysis) -> None:
    rows = sorted((s for s in a.signals if s.counts), key=lambda s: s.score * s.weight, reverse=True)
    items = []
    for s in rows:
        icon, label, color = _status(s.score)
        items.append(
            f'<div class="crit"><span>{icon}</span><span class="c-name">{html.escape(s.name)}</span>'
            f'<span class="c-badge" style="--b:{color}">{label} · {s.score:.0%}</span>'
            f'<span class="c-detail">{html.escape(s.detail)}</span></div>'
        )
    st.markdown("".join(items), unsafe_allow_html=True)
    skipped = [s.name for s in a.signals if not s.counts]
    if skipped:
        st.caption("No evidence either way: " + ", ".join(skipped))


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])
