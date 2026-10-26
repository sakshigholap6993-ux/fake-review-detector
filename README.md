# 🕵️ Fake Review Detector

A Streamlit web app that checks whether a product review looks **fake** or **real**.
Paste the review text, or upload a screenshot and the app reads the text with OCR.
Every verdict comes with a breakdown of *why*.

## Features

- **Two inputs**: paste text, or upload a screenshot (PNG/JPG/WEBP). The OCR text can be edited before analysis.
- **Verdict + confidence**: FAKE / REAL with a 0–100% fake score, plus a warning when the call is borderline.
- **Explainable criteria**: each rule shows its score and reasoning, and the phrases that triggered it are highlighted in the review.
- **Screenshot extras**: the star rating ("4.0 out of 5") and the "Verified Purchase" badge are picked up automatically.
- **One-click examples** for demos: each one fills the box and runs the analysis right away.
- **Interactive results**: an animated verdict card with a score meter, plus four views. *Highlights* marks phrases in colors by criterion. *Criteria* is a bar chart with hover explanations. *Sentence mood* charts the tone of each sentence. *Details* lists every rule.
- **Strictness control** in the sidebar (Relaxed / Balanced / Strict): Strict flags more reviews. Changing it updates the verdict immediately, without re-running the analysis.
- **Session history** table of every review checked in this session.
- Follows the viewer's **light/dark** theme and works on phones.

## How it works

Two detectors run on every review. A review is flagged if **either** one is confident, because they catch different kinds of fake:

| Detector | Catches | How |
|---|---|---|
| **ML model** | Machine-generated reviews | TF-IDF (word + character n-grams) + logistic regression, trained on 40k reviews. Held-out accuracy **95.1%**, ROC AUC **0.989**. |
| **Rule checks** | Human-written paid, hyped or spam reviews | 11 explainable criteria combined as a weighted average. A damning signal (spam link, paid-review disclosure, rating mismatch) sets a minimum score, so it can't be averaged away |

### Rule criteria

| Criterion | Why it matters |
|---|---|
| Extreme / superlative words | Fakes lean on "amazing", "perfect", "worst" |
| Stock / template phrases | Paid reviews reuse "highly recommend", "worth every penny" |
| Excessive punctuation / CAPS | Over-the-top emphasis ("BEST EVER!!!") |
| Self-focused language | Deceptive reviews talk about the writer more than the product (Ott et al., 2011) |
| Lack of concrete details | Genuine buyers mention features, numbers and how long they used it |
| One-sided sentiment | Real experiences usually mention some trade-off |
| Very short review | Little evidence of real use |
| Repetitive wording | Bot / copy-paste text repeats itself |
| Links / promo codes / contact info* | Advertising inside a review is spam |
| Incentivized disclosure* | "Received free in exchange for an honest review" |
| Star rating vs. text mismatch* | A 5-star rating on a complaint (or the reverse) |

\* *Evidence-only*: these count when they fire. When absent, they don't pull the score toward "real".

## Project structure

```
app.py                  Streamlit page (layout, state, history)
ui/
  components.py         verdict card, tiles, highlights, Plotly charts
  style.css             gradient hero, cards, animations, light/dark colors
detector/
  analyzer.py           combines ML + rules into the verdict
  heuristics.py         the rule-based criteria
  model.py              loads the trained classifier
  ocr.py                screenshot -> review text (RapidOCR)
models/review_model.joblib   trained model (2 MB, committed)
train_model.py          retrains the model
samples/                example screenshot for demos
tests/                  pytest suite
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. To run the tests: `pip install pytest`, then `pytest`.

### Retrain the model (optional)

The trained model is already in `models/`. To rebuild it, run the script below. It trains on the dataset in `data/` (included in the repo; see [data/README.md](data/README.md) for source and licence).

```bash
python train_model.py
```

You can also train on your own labelled CSV:
`python train_model.py --data my.csv --text-col review --label-col is_fake --fake-label 1`

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repository**, with `models/review_model.joblib` included.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click **Create app**.
3. Pick the repo and branch, and set the main file path to `app.py`.
4. Under **Advanced settings**, choose **Python 3.11**, the version the model was trained with.
5. Click **Deploy**. You get a public `https://<your-app>.streamlit.app` URL to share.

No secrets or API keys are needed. The OCR models ship inside the `rapidocr` package, and `opencv-python-headless` needs no system packages.

> Note: `scikit-learn` is pinned because the saved model only loads with the version that trained it. If you upgrade it, re-run `train_model.py`.

## Limitations

- The ML model learned from **computer-generated** fakes. Human-written fakes are caught by the rule checks, which are heuristics rather than proof.
- English only. OCR works best on clear, cropped screenshots.
- Scores are probabilistic. Don't use them as the only basis for decisions about real people or businesses.

## Credits

- Dataset: Salminen, J., Kandpal, C., Kamel, A. M., Jung, S., & Jansen, B. J. (2022). *Creating and detecting fake reviews of online products.* Journal of Retailing and Consumer Services, 64. Data: <https://osf.io/tyue9/>
- Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). *Finding Deceptive Opinion Spam by Any Stretch of the Imagination.* ACL.
- Sentiment: [VADER](https://github.com/cjhutto/vaderSentiment) · OCR: [RapidOCR](https://github.com/RapidAI/RapidOCR)
