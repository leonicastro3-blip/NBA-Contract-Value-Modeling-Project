"""Download raw data, pinned to exact commits so results are reproducible.

Sources
-------
- Basketball-Reference season tables, compiled by Sumitro Datta
  (github.com/sumitrodatta/bball-reference-datasets; also on Kaggle as
  "NBA Stats (1947-present)").
- Player salaries 2009-10 through 2024-25 (HoopsHype-derived, via
  github.com/tripledipple/nba-salary-prediction).
- Player salaries 2025-26 (github.com/zalii00/nba-salary-analysis).
- Player salaries and active rosters 2026-27: Spotrac team cash tables captured
  2026-10-09 (github.com/jesusguy115-crypto/nba-take-home-pay).

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

BBREF = "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/76a70b41ad1c13948f25c62c921ed822e5db7f0e/Data/"
SAL_HIST = "https://raw.githubusercontent.com/tripledipple/nba-salary-prediction/833809578b7cdf47c81ee7bca64630897339ba54/"
SAL_2026 = "https://raw.githubusercontent.com/zalii00/nba-salary-analysis/e1d4553e91736ea6bc7c4004ee5a8d47d37b0791/"
SAL_2027 = ("https://raw.githubusercontent.com/jesusguy115-crypto/nba-take-home-pay/"
            "87f7c1f63d66caeffe8f41d62879e43dda2d5d83/skills/nba-take-home-pay/assets/")

FILES = {
    "advanced.csv": BBREF + "Advanced.csv",
    "per_game.csv": BBREF + "Player Per Game.csv",
    "season_info.csv": BBREF + "Player Season Info.csv",
    "draft_history.csv": BBREF + "Draft Pick History.csv",
    "all_star.csv": BBREF + "All-Star Selections.csv",
    "team_summaries.csv": BBREF + "Team Summaries.csv",
    "salaries_2010_2025.csv": SAL_HIST + "NBA_Player_Stats_and_Salaries_2010-2025.csv",
    "salaries_2026.csv": SAL_2026 + "salaries_2025_26.csv",
    "salaries_2027.json": SAL_2027 + "salaries-2026-27.json",
}


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"  exists   {name}")
            continue
        safe_url = urllib.parse.quote(url, safe=":/")
        print(f"  fetching {name}")
        urllib.request.urlretrieve(safe_url, dest)
    print(f"Raw data in {RAW_DIR}")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
