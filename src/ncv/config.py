"""Project-wide configuration: paths, league constants, and modeling settings.

Season convention: an integer season refers to the year the season ENDS,
matching Basketball-Reference (2026 == the 2025-26 season).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Official NBA salary cap by season (season = ending year).
# Sources: NBA.com press releases; cross-checked against an independent dataset.
SALARY_CAP: dict[int, int] = {
    2010: 57_700_000,
    2011: 58_044_000,
    2012: 58_044_000,
    2013: 58_044_000,
    2014: 58_679_000,
    2015: 63_065_000,
    2016: 70_000_000,
    2017: 94_143_000,
    2018: 99_093_000,
    2019: 101_869_000,
    2020: 109_140_000,
    2021: 109_140_000,
    2022: 112_414_000,
    2023: 123_655_000,
    2024: 136_021_000,
    2025: 140_588_000,
    2026: 154_647_000,
    2027: 164_961_000,
}

# Basketball-Reference: VORP * 2.7 approximates wins above replacement.
VORP_TO_WINS = 2.7

# First-round picks sign 4-year rookie-scale deals (years 3-4 are team options).
ROOKIE_SCALE_YEARS = 4


@dataclass(frozen=True)
class ModelSettings:
    first_stat_season: int = 2010   # earliest season with salary data
    last_salary_season: int = 2026  # latest season with salaries (2025-26)
    backtest_seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)
    min_minutes_lag: int = 250      # minimum prior-season minutes to be forecast
    min_minutes_value: int = 500    # rotation threshold: value boards, replacement salary, archetypes

    # Availability. A season is "injury-shortened" when a player appeared in fewer than
    # `injury_games_pct` of his team's games AND was a rotation player (>= `injury_min_mpg`
    # minutes in the games he played, or >= `injury_prior_minutes` the season before).
    # Games played cannot separate injury from trade, suspension or benching; the flag
    # means "missed most of the season", not "was injured".
    injury_games_pct: float = 0.5
    injury_min_mpg: float = 15.0
    injury_prior_minutes: int = 1000
    # Healthy-pace value: expected wins in missed games, from his per-game rate shrunk
    # toward his prior-season rate (or the league rotation average) with this many games
    # of prior weight. Rates need at least `min_games_for_rate` games to be trusted.
    healthy_rate_prior_games: int = 20
    min_games_for_rate: int = 20
    forecast_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_archetypes: int = 8
    random_state: int = 42
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
    })


SETTINGS = ModelSettings()


def season_label(season: int) -> str:
    """2026 -> '2025-26'."""
    return f"{season - 1}-{str(season)[-2:]}"
