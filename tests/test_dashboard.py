"""Smoke test: the dashboard renders and every widget interaction runs without error.
Skipped until `python -m ncv.pipeline` has produced outputs."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/projections_2027.parquet").exists(),
                                reason="run the pipeline first")


def test_dashboard_interactions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5
    roster = at.dataframe[0].value
    assert len(roster) >= 8 and "Luka Dončić" in set(roster["Player"])
    at.sidebar.selectbox[0].set_value("OKC").run()
    assert not at.exception and len(at.dataframe[0].value) >= 8
    at.checkbox[0].check().run()
    at.multiselect[0].set_value(["C"]).run()
    assert not at.exception
    at.radio[0].set_value("Market value (track record)").run()
    at.selectbox[0].set_value("Victor Wembanyama").run()
    assert not at.exception
