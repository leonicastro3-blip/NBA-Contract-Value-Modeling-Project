import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ncv.data.panel import LAG1_COLS  # noqa: E402


@pytest.fixture
def season_table() -> pd.DataFrame:
    """Small synthetic player-season table with every column the panel needs."""
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(60):
        start = 2008 + pid % 5
        for i, season in enumerate(range(start, 2019)):
            r = {c: rng.normal(1, 0.5) for c in LAG1_COLS}
            r.update(player_id=f"p{pid}", player=f"Player {pid}", season=season, team="AAA",
                     pos=["PG", "SG", "SF", "PF", "C"][pid % 5], age=21 + i, experience=i + 1,
                     draft_round=1 if pid % 3 == 0 else (2 if pid % 3 == 1 else np.nan),
                     draft_pick=float(pid % 60 + 1), all_star=0.0,
                     mp=float(rng.integers(300, 2800)), g=float(rng.integers(20, 82)),
                     gs=float(rng.integers(0, 20)), war=rng.normal(2, 2),
                     team_games=82.0, injury_shortened=0)
            r["vorp"] = r["war"] / 2.7
            r["salary"] = 1e6 + max(0, r["war"]) * 3e6 + rng.normal(0, 5e5)
            r["salary_cap"] = 1e8
            r["pct_cap"] = r["salary"] / r["salary_cap"]
            rows.append(r)
    st = pd.DataFrame(rows)
    st["is_rookie_scale"] = (st["draft_round"] == 1) & (st["experience"] <= 4)
    return st
