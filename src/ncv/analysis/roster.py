"""Roster-construction outputs built on top of the three models.

Vocabulary used throughout (all as share of the salary cap unless noted):
  market_value        what the open market would pay for the player's track
                      record entering the season (ex-ante, out-of-sample)
  production_value    what the player's realized wins were worth (ex-post)
  market_surplus      market_value - salary   (contract vs going rate)
  realized_surplus    production_value - salary  (contract vs what he delivered)
  projected_value_pXX value of the P10/P50/P90 WAR forecast for next season
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ncv.config import ROOT, SALARY_CAP
from ncv.models.production import production_value

ROSTER_DIR = ROOT / "data" / "rosters"
BOARD_COLS = [
    "player", "player_id", "team", "pos", "age", "archetype", "contract_type",
    "salary", "pct_cap", "market_value", "market_surplus", "production_value",
    "realized_surplus", "war", "mp", "g", "team_games", "games_pct", "injury_shortened",
    "healthy_value", "healthy_surplus", "availability_cost", "drivers",
]


def contract_type(is_rookie_scale: pd.Series) -> pd.Series:
    return np.where(is_rookie_scale, "Rookie scale", "Veteran")


def current_board(market_oof: pd.DataFrame, season_table: pd.DataFrame, prices: pd.DataFrame,
                  season: int, drivers: pd.Series | None = None) -> pd.DataFrame:
    """Value board for a completed season using out-of-sample market values."""
    b = market_oof[market_oof["season"] == season].copy()
    b = b.merge(season_table[["player_id", "season", "pos", "archetype", "team_games", "games_pct",
                              "injury_shortened", "healthy_war"]]
                .rename(columns={"pos": "pos_T"}), on=["player_id", "season"], how="left")
    b["pos"] = b["pos_T"].fillna(b["pos"])
    b["war"], b["mp"], b["g"] = b["war_T"], b["mp_T"], b["g_T"]
    b["production_value"] = production_value(b["war"].fillna(0), b["season"], prices)
    # Healthy-pace value: what the player's wins would have been worth over the games he missed
    # too. availability_cost = healthy - realized, zero for a full season by construction.
    b["healthy_value"] = production_value(b["healthy_war"].fillna(b["war"]).fillna(0), b["season"], prices)
    b["availability_cost"] = (b["healthy_value"] - b["production_value"]).clip(lower=0)
    b["healthy_surplus"] = b["healthy_value"] - b["pct_cap"]
    b["injury_shortened"] = b["injury_shortened"].fillna(0).astype(int)
    b["market_surplus"] = b["market_value"] - b["pct_cap"]
    b["realized_surplus"] = b["production_value"] - b["pct_cap"]
    b["contract_type"] = contract_type(b["is_rookie_scale"])
    b["drivers"] = drivers.reindex(b.index) if drivers is not None else ""
    return b[BOARD_COLS].sort_values("realized_surplus", ascending=False).reset_index(drop=True)


def load_contracts(season: int) -> pd.DataFrame | None:
    """Optional actual salaries for a future season: data/rosters/contracts_<season>.csv
    with columns player,salary. Returns None if the file is absent."""
    p = ROSTER_DIR / f"contracts_{season}.csv"
    return pd.read_csv(p) if p.exists() else None


def projection_board(features_next: pd.DataFrame, quantiles: pd.DataFrame, market_value: np.ndarray,
                     prices: pd.DataFrame, season_table: pd.DataFrame, season: int,
                     contracts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Next-season projections with uncertainty and a cost basis.

    Price of a win: average of the last three seasons' cap share per WAR.
    Cost basis, in priority order:
      1. data/rosters/contracts_<season>.csv (manual override)
      2. `contracts` snapshot (actual next-season salaries + current teams)
      3. prior-season share of cap (proxy)
    Players absent from a supplied snapshot are labeled as not on a roster.
    """
    from ncv.data.cleaning import name_key

    recent = prices[prices["season"].between(season - 3, season - 1)]
    rep, per_war = recent["rep_pct_cap"].mean(), recent["pct_cap_per_war"].mean()

    b = features_next[["player", "player_id", "team_prev", "pos", "age", "experience",
                       "is_rookie_scale", "draft_round"]].copy()
    b = b.join(quantiles)
    for c in quantiles.columns:
        b[c.replace("war_", "value_")] = (rep + b[c] * per_war).clip(lower=0)
    b["market_value"] = np.clip(market_value, 0, None)

    prev = season_table[season_table["season"] == season - 1][["player_id", "pct_cap", "archetype"]]
    b = b.merge(prev, on="player_id", how="left")
    b["k"] = b["player"].map(name_key)
    b["team"] = b["team_prev"]
    b["cost_basis"] = np.where(b["pct_cap"].notna(), "prior-season salary (proxy)", "unknown")
    rookie_done = (b["draft_round"] == 1) & (b["experience"] == 5)
    b.loc[rookie_done, "cost_basis"] = "rookie deal ended (new contract)"

    if contracts is not None:
        snap = contracts[["name_key", "salary", "team"]].rename(
            columns={"name_key": "k", "salary": "_sal", "team": "_team"})
        b = b.merge(snap, on="k", how="left")
        found = b["_sal"].notna()
        b.loc[found, "pct_cap"] = b.loc[found, "_sal"] / SALARY_CAP[season]
        b.loc[found, "cost_basis"] = f"actual salary ({contracts.attrs.get('captured_date', 'snapshot')})"
        b.loc[found & b["_team"].notna(), "team"] = b.loc[found & b["_team"].notna(), "_team"]
        b.loc[found & b["_team"].isna(), "team"] = "Dead money only"
        b.loc[~found, "cost_basis"] = "not on a roster"
        b.loc[~found, "team"] = "Unsigned"
        b = b.drop(columns=["_sal", "_team"])

    manual = load_contracts(season)
    if manual is not None:
        m = manual.assign(k=manual["player"].map(name_key))[["k", "salary"]].rename(columns={"salary": "_m"})
        b = b.merge(m, on="k", how="left")
        has = b["_m"].notna()
        b.loc[has, "pct_cap"] = b.loc[has, "_m"] / SALARY_CAP[season]
        b.loc[has, "cost_basis"] = "actual salary (manual)"
        b = b.drop(columns="_m")

    priced = b["cost_basis"].str.startswith("actual") | (b["cost_basis"] == "prior-season salary (proxy)")
    for q in ["p10", "p50", "p90"]:
        b[f"surplus_{q}"] = np.where(priced, b[f"value_{q}"] - b["pct_cap"], np.nan)
    b["contract_type"] = contract_type(b["is_rookie_scale"])
    b["salary_cap"] = SALARY_CAP[season]
    return b.drop(columns="k").sort_values("war_p50", ascending=False).reset_index(drop=True)


def team_summary(board: pd.DataFrame, team_records: pd.DataFrame) -> pd.DataFrame:
    """Team payroll efficiency for a completed season, with win totals."""
    paid = board[board["salary"].notna()]
    t = paid.groupby("team").agg(
        payroll_tracked=("salary", "sum"), war=("war", "sum"),
        realized_surplus=("realized_surplus", "sum"), market_surplus=("market_surplus", "sum"),
        rookie_scale_war=("war", lambda w: w[paid.loc[w.index, "contract_type"] == "Rookie scale"].sum()),
        n_players=("player", "size"))
    t["usd_per_war"] = t["payroll_tracked"] / t["war"].clip(lower=1)
    t = t.join(team_records.set_index("team"))
    return t.sort_values("realized_surplus", ascending=False).reset_index()


def find_targets(proj: pd.DataFrame, min_salary: float = 0, max_salary: float = 1e9,
                 positions: list[str] | None = None, max_age: float = 40,
                 require_positive_p10: bool = False, exclude_team: str | None = None) -> pd.DataFrame:
    """Trade/free-agent screen on projected surplus, with salary-matching bands in dollars."""
    p = proj.copy()
    dollars = p["pct_cap"] * p["salary_cap"]
    keep = dollars.between(min_salary, max_salary) & (p["age"] <= max_age) & p["surplus_p50"].notna()
    if positions:
        keep &= p["pos"].isin(positions)
    if require_positive_p10:
        keep &= p["surplus_p10"] > 0
    if exclude_team:
        keep &= p["team"] != exclude_team
    return p[keep].sort_values("surplus_p50", ascending=False)


def read_roster(path: Path) -> list[str]:
    return pd.read_csv(path)["player"].tolist()
