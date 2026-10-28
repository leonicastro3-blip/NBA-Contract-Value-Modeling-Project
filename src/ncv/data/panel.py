"""Build the modeling panel.

Every modeling row is a (player, target season T) pair. Features come only
from seasons T-1 and T-2, plus deterministic facts known before T (age,
experience, draft slot). That makes the same feature frame valid for:
  * the market-value model (what the open market pays entering season T), and
  * the forecast model (what the player produces in season T).
Targets (salary in T, production in T) are joined separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ncv.config import ROOKIE_SCALE_YEARS, SALARY_CAP, SETTINGS, VORP_TO_WINS
from ncv.data import sources

LAG1_COLS = [
    "g", "gs", "mp", "mp_per_game", "per", "ts_percent", "usg_percent", "ast_percent",
    "trb_percent", "orb_percent", "drb_percent", "stl_percent", "blk_percent", "tov_percent",
    "x3p_ar", "f_tr", "obpm", "dbpm", "bpm", "vorp", "ws", "ws_48",
    "pts_per_game", "ast_per_game", "trb_per_game", "x3p_percent", "ft_percent",
]
LAG2_COLS = ["g", "mp", "bpm", "obpm", "dbpm", "vorp", "ws", "usg_percent", "pts_per_game", "ts_percent"]
POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def add_availability(st: pd.DataFrame) -> pd.DataFrame:
    """Availability columns: share of team games played, an injury-shortened flag, and
    healthy-pace WAR (realized WAR plus expected WAR over the games he missed).

    Expected WAR in missed games uses his per-game rate shrunk toward his prior-season
    rate (or the league rotation average if he has no usable prior season), floored at
    zero: a below-replacement player's missed games would have been played by a
    replacement-level player, so missing them never costs wins. A full season therefore
    has healthy_war == war and an availability cost of exactly zero.
    """
    S = SETTINGS
    st = st.copy()
    st["games_pct"] = (st["g"] / st["team_games"]).clip(upper=1.0)

    prev = st[["player_id", "season", "mp", "g", "war"]].copy()
    prev["season"] = prev["season"] + 1
    prev = prev.rename(columns={"mp": "prior_mp", "g": "prior_g", "war": "prior_war"})
    st = st.merge(prev, on=["player_id", "season"], how="left", validate="one_to_one")

    rotation_pace = st["mp_per_game"] >= S.injury_min_mpg
    full_prior = st["prior_mp"].fillna(0) >= S.injury_prior_minutes
    st["injury_shortened"] = ((st["games_pct"] < S.injury_games_pct)
                              & (rotation_pace | full_prior)).astype(int)

    st["war_per_game"] = st["war"] / st["g"]
    usable_prior = st["prior_g"].fillna(0) >= S.min_games_for_rate
    st["prior_war_per_game"] = np.where(usable_prior, st["prior_war"] / st["prior_g"], np.nan)
    rotation = st[rotation_pace & (st["g"] >= S.min_games_for_rate)]
    st["league_war_per_game"] = st["season"].map(rotation.groupby("season")["war_per_game"].mean())

    prior_rate = st["prior_war_per_game"].fillna(st["league_war_per_game"])
    k = S.healthy_rate_prior_games
    shrunk = (st["g"] * st["war_per_game"] + k * prior_rate) / (st["g"] + k)
    st["healthy_rate"] = shrunk.clip(lower=0)
    st["games_missed"] = (st["team_games"] - st["g"]).clip(lower=0)
    st["healthy_war"] = st["war"] + st["healthy_rate"] * st["games_missed"]
    return st.drop(columns=["prior_mp", "prior_g", "prior_war"])


def load_season_table() -> tuple[pd.DataFrame, dict]:
    """Player-season table with stats, salary, draft info, All-Star flags, availability."""
    stats = sources.load_stats()
    stats = sources.attach_team_games(stats)
    stats, diag = sources.attach_salaries(stats, sources.load_salaries())
    stats = stats.merge(sources.load_draft(), on="player_id", how="left")
    stats = stats.merge(sources.load_all_stars(), on=["player_id", "season"], how="left")
    stats["all_star"] = stats["all_star"].fillna(0).clip(upper=1)
    stats["war"] = stats["vorp"] * VORP_TO_WINS
    stats["is_rookie_scale"] = is_rookie_scale(stats["draft_round"], stats["experience"])
    stats = add_availability(stats)
    return stats, diag


def is_rookie_scale(draft_round: pd.Series, experience: pd.Series) -> pd.Series:
    """First-round picks in their first four NBA seasons are on CBA-set rookie scale."""
    return (draft_round == 1) & (experience <= ROOKIE_SCALE_YEARS)


def build_features(season_table: pd.DataFrame, target_seasons: list[int]) -> pd.DataFrame:
    """One row per (player, T) for players who logged minutes in T-1.

    Strictly uses data from seasons < T (verified in tests/test_panel.py).
    """
    s = season_table.sort_values(["player_id", "season"])
    lag1 = s[["player_id", "season", "player", "team", "pos", "age", "experience",
              "draft_round", "draft_pick", "all_star", "team_games", "injury_shortened"]
             + LAG1_COLS].copy()
    lag1 = lag1.rename(columns={c: f"l1_{c}" for c in
                                LAG1_COLS + ["all_star", "team_games", "injury_shortened"]})
    lag1 = lag1.rename(columns={"team": "team_prev"})
    lag1["season"] = lag1["season"] + 1

    lag2 = s[["player_id", "season"] + LAG2_COLS + ["all_star"]].copy()
    lag2 = lag2.rename(columns={c: f"l2_{c}" for c in LAG2_COLS + ["all_star"]})
    lag2["season"] = lag2["season"] + 2

    # Career context strictly before T: cumulative VORP and All-Star count
    s = s.assign(cum_vorp=s.groupby("player_id")["vorp"].cumsum(),
                 cum_mp=s.groupby("player_id")["mp"].cumsum(),
                 cum_as=s.groupby("player_id")["all_star"].cumsum())
    career = s[["player_id", "season", "cum_vorp", "cum_mp", "cum_as"]].copy()
    career["season"] = career["season"] + 1
    career = career.rename(columns={"cum_vorp": "career_vorp_prior",
                                    "cum_mp": "career_mp_prior",
                                    "cum_as": "career_all_stars_prior"})

    f = lag1[lag1["season"].isin(target_seasons)]
    f = f.merge(lag2, on=["player_id", "season"], how="left")
    # Keep regular rotation players, plus injury-shortened seasons preceded by a
    # full one (e.g. a 154-minute season after 1,740 minutes). Without the second
    # condition the model never learns how players return from major injuries.
    f = f[(f["l1_mp"] >= SETTINGS.min_minutes_lag) | (f["l2_mp"] >= SETTINGS.injury_prior_minutes)]
    f = f.merge(career, on=["player_id", "season"], how="left")

    f["age"] = f["age"] + 1                      # age during season T
    f["experience"] = f["experience"] + 1        # experience during season T
    f["draft_pick"] = f["draft_pick"].fillna(61)  # undrafted = after last pick
    f["is_rookie_scale"] = is_rookie_scale(f["draft_round"], f["experience"])
    f["undrafted"] = f["draft_round"].isna().astype(int)

    # Minutes-weighted two-year impact (stabilizes noisy single seasons)
    for m in ["bpm", "obpm", "dbpm"]:
        w1, w2 = f["l1_mp"], f["l2_mp"].fillna(0)
        f[f"wt_{m}"] = (f[f"l1_{m}"] * w1 + f[f"l2_{m}"].fillna(0) * w2) / (w1 + w2)
    f["l1_gs_share"] = f["l1_gs"] / f["l1_g"].replace(0, np.nan)
    f["l1_missed_games"] = (f["l1_team_games"] - f["l1_g"]).clip(lower=0)
    f["l1_games_pct"] = (f["l1_g"] / f["l1_team_games"]).clip(upper=1.0)
    f["played_l2"] = f["l2_mp"].notna().astype(int)
    for p in POSITIONS:
        f[f"pos_{p}"] = (f["pos"] == p).astype(int)
    return f.reset_index(drop=True)


def attach_targets(features: pd.DataFrame, season_table: pd.DataFrame) -> pd.DataFrame:
    """Join season-T outcomes: salary/pct_cap (market) and WAR/minutes (forecast)."""
    t = season_table[["player_id", "season", "team", "salary", "pct_cap", "war", "vorp",
                      "bpm", "mp", "g"]].rename(columns={
        "team": "team_T", "war": "war_T", "vorp": "vorp_T", "bpm": "bpm_T",
        "mp": "mp_T", "g": "g_T"})
    out = features.merge(t, on=["player_id", "season"], how="left", validate="one_to_one")
    out["salary_cap"] = out["season"].map(SALARY_CAP)
    out["team"] = out["team_T"].fillna(out["team_prev"])
    return out


FEATURES: list[str] = (
    ["age", "experience", "draft_pick", "undrafted"]
    + [f"l1_{c}" for c in LAG1_COLS] + ["l1_all_star", "l1_gs_share", "l1_missed_games", "l1_games_pct",
       "l1_injury_shortened"]
    + [f"l2_{c}" for c in LAG2_COLS] + ["l2_all_star", "played_l2"]
    + ["wt_bpm", "wt_obpm", "wt_dbpm"]
    + ["career_vorp_prior", "career_mp_prior", "career_all_stars_prior"]
    + [f"pos_{p}" for p in POSITIONS]
)

# Readable names for plots and the dashboard
FEATURE_LABELS = {
    "l1_mp": "Minutes (last season)", "l1_vorp": "VORP (last season)",
    "l1_bpm": "BPM (last season)", "wt_bpm": "BPM (2-yr, minutes-weighted)",
    "wt_obpm": "Offensive BPM (2-yr)", "wt_dbpm": "Defensive BPM (2-yr)",
    "l1_pts_per_game": "Points per game (last season)", "l1_usg_percent": "Usage rate (last season)",
    "l1_ws": "Win Shares (last season)", "l1_gs": "Games started (last season)",
    "l1_gs_share": "Share of games started", "l1_mp_per_game": "Minutes per game (last season)",
    "age": "Age", "experience": "Years of experience", "draft_pick": "Draft slot",
    "career_vorp_prior": "Career VORP to date", "career_all_stars_prior": "Career All-Star selections",
    "career_mp_prior": "Career minutes to date", "l1_all_star": "All-Star last season",
    "l2_vorp": "VORP (two seasons ago)", "l2_mp": "Minutes (two seasons ago)",
    "l1_per": "PER (last season)", "l1_ts_percent": "True shooting % (last season)",
    "l1_ast_percent": "Assist rate (last season)", "l1_ws_48": "WS/48 (last season)",
    "l2_pts_per_game": "Points per game (two seasons ago)", "l1_g": "Games played (last season)",
    "l1_missed_games": "Team games missed (last season)",
    "l1_games_pct": "Share of team games played (last season)",
    "l1_injury_shortened": "Injury-shortened last season",
}


_STAT_NAMES = {
    "g": "Games played", "gs": "Games started", "mp": "Minutes", "mp_per_game": "Minutes per game",
    "per": "PER", "ts_percent": "True shooting %", "usg_percent": "Usage rate",
    "ast_percent": "Assist rate", "trb_percent": "Rebound rate", "orb_percent": "Off. rebound rate",
    "drb_percent": "Def. rebound rate", "stl_percent": "Steal rate", "blk_percent": "Block rate",
    "tov_percent": "Turnover rate", "x3p_ar": "3-point attempt rate", "f_tr": "Free-throw rate",
    "obpm": "Offensive BPM", "dbpm": "Defensive BPM", "bpm": "BPM", "vorp": "VORP",
    "ws": "Win Shares", "ws_48": "WS/48", "pts_per_game": "Points per game",
    "ast_per_game": "Assists per game", "trb_per_game": "Rebounds per game",
    "x3p_percent": "3-point %", "ft_percent": "Free-throw %", "all_star": "All-Star",
}


def label(feature: str) -> str:
    """Readable feature name; generated for any lagged stat not listed explicitly."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for prefix, when in (("l1_", "last season"), ("l2_", "two seasons ago")):
        if feature.startswith(prefix) and feature[3:] in _STAT_NAMES:
            return f"{_STAT_NAMES[feature[3:]]} ({when})"
    if feature.startswith("pos_"):
        return f"Position: {feature[4:]}"
    return {"undrafted": "Undrafted", "played_l2": "Played two seasons ago"}.get(feature, feature)
