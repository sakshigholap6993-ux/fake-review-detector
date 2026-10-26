"""Smoke test: run the whole Streamlit page headless, the way Streamlit Cloud will."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_loads_and_analyzes_example():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception

    at.button[0].click().run()          # "Suspicious" example fills the box and auto-analyzes
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Likely FAKE" in html
    assert len(at.dataframe) == 1       # session history recorded the check
