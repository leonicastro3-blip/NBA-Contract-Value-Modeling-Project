"""Production-value lens: what a player's on-court output was worth.

Wins above replacement (WAR = VORP x 2.7) priced at the league's own
marginal cost of a win that season:

    $/WAR_t   = sum(salary - replacement salary)+ / sum(WAR)+
    value_i,t = replacement salary + WAR_i,t x $/WAR_t     (floored at 0)

Replacement salary is the 10th percentile salary among rotation players
(>= `min_minutes_value` minutes), which tracks the minimum-salary band at ~1.4% of the cap.
Everything is expressed as a share of that season's cap.
"""
from __future__ import annotations

import pandas as pd

from ncv.config import SETTINGS

ROTATION_MINUTES = SETTINGS.min_minutes_value


def league_win_prices(season_table: pd.DataFrame) -> pd.DataFrame:
    """Per-season replacement salary and marginal price of a win (share of cap)."""
    paid = season_table[season_table["salary"].notna()]
    rep = (paid[paid["mp"] >= ROTATION_MINUTES].groupby("season")["pct_cap"]
           .quantile(0.10).rename("rep_pct_cap"))
    paid = paid.join(rep, on="season")
    agg = paid.groupby("season").apply(
        lambda d: pd.Series({
            "marginal_spend": (d["pct_cap"] - d["rep_pct_cap"]).clip(lower=0).sum(),
            "war_pos": d["war"].clip(lower=0).sum(),
        }), include_groups=False)
    prices = rep.to_frame().join(agg)
    prices["pct_cap_per_war"] = prices["marginal_spend"] / prices["war_pos"]
    return prices.reset_index()


def production_value(war: pd.Series, season: pd.Series, prices: pd.DataFrame) -> pd.Series:
    """Value (share of cap) of a given WAR in a given season's price environment."""
    p = prices.set_index("season")
    rep = season.map(p["rep_pct_cap"])
    per_war = season.map(p["pct_cap_per_war"])
    return (rep + war * per_war).clip(lower=0)
