"""Load raw sources into tidy player-season tables."""
from __future__ import annotations

import logging

import pandas as pd

from ncv.config import RAW_DIR, SALARY_CAP, SETTINGS
from ncv.data.cleaning import collapse_multi_team, name_key

log = logging.getLogger(__name__)

# Spotrac abbreviations -> Basketball-Reference abbreviations
TEAM_CODE_FIXES = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}

ADV_COLS = [
    "per", "ts_percent", "x3p_ar", "f_tr", "orb_percent", "drb_percent", "trb_percent",
    "ast_percent", "stl_percent", "blk_percent", "tov_percent", "usg_percent",
    "ows", "dws", "ws", "ws_48", "obpm", "dbpm", "bpm", "vorp",
]
PG_COLS = [
    "mp_per_game", "fg_per_game", "fga_per_game", "fg_percent", "x3p_per_game",
    "x3pa_per_game", "x3p_percent", "ft_per_game", "fta_per_game", "ft_percent",
    "orb_per_game", "drb_per_game", "trb_per_game", "ast_per_game", "stl_per_game",
    "blk_per_game", "tov_per_game", "pf_per_game", "pts_per_game", "e_fg_percent",
]


def load_stats(first_season: int = 2007) -> pd.DataFrame:
    """Advanced + per-game stats, one row per NBA player-season."""
    adv = pd.read_csv(RAW_DIR / "advanced.csv")
    pg = pd.read_csv(RAW_DIR / "per_game.csv")
    adv = adv[(adv["lg"] == "NBA") & (adv["season"] >= first_season)]
    pg = pg[(pg["lg"] == "NBA") & (pg["season"] >= first_season)]

    adv = collapse_multi_team(adv)
    pg = collapse_multi_team(pg)

    base = ["season", "player", "player_id", "age", "team", "pos", "g", "gs", "mp", "n_teams"]
    stats = adv[base + ADV_COLS].merge(
        pg[["player_id", "season"] + PG_COLS], on=["player_id", "season"], how="left",
        validate="one_to_one",
    )

    info = pd.read_csv(RAW_DIR / "season_info.csv")
    info = info[info["lg"] == "NBA"].drop_duplicates(["player_id", "season"])
    stats = stats.merge(info[["player_id", "season", "experience"]],
                        on=["player_id", "season"], how="left", validate="one_to_one")

    stats["pos"] = stats["pos"].astype(str).str.split("-").str[0]  # 'SF-PF' -> 'SF'
    stats["name_key"] = stats["player"].map(name_key)
    return stats


def load_draft() -> pd.DataFrame:
    d = pd.read_csv(RAW_DIR / "draft_history.csv")
    d = d[(d["lg"] == "NBA") & d["player_id"].notna()]
    d = d.sort_values("season").drop_duplicates("player_id", keep="last")
    return d.rename(columns={"season": "draft_season", "overall_pick": "draft_pick",
                             "round": "draft_round"})[
        ["player_id", "draft_season", "draft_pick", "draft_round"]]


def load_all_stars() -> pd.DataFrame:
    a = pd.read_csv(RAW_DIR / "all_star.csv")
    a = a[a["lg"] == "NBA"]
    return a.groupby(["player_id", "season"]).size().rename("all_star").reset_index()


def load_team_records() -> pd.DataFrame:
    t = pd.read_csv(RAW_DIR / "team_summaries.csv")
    t = t[(t["lg"] == "NBA") & t["abbreviation"].notna()]
    t = t.drop(columns="team").rename(columns={"abbreviation": "team"})
    return t[["season", "team", "w", "l", "n_rtg"]]


def load_next_season_contracts() -> pd.DataFrame | None:
    """2026-27 salaries and active-roster teams (Spotrac snapshot, 2026-10-09).

    Returns player, name_key, salary (total cash owed for the season, including
    dead money from a former team) and team (active roster at capture; None if
    the player is only owed dead money). None if the file is absent.
    """
    import json
    path = RAW_DIR / "salaries_2027.json"
    if not path.exists():
        return None
    snap = json.loads(path.read_text())
    rows = [{"player": p["player"], "salary": p["cash_total_usd"],
             "team": (p["teams_in_active_roster_sections"] or [None])[-1]} for p in snap["players"]]
    df = pd.DataFrame(rows)
    df["team"] = df["team"].replace(TEAM_CODE_FIXES)
    df["name_key"] = df["player"].map(name_key)
    df.attrs["captured_date"] = snap["captured_date"]
    return df.drop_duplicates("name_key")


def attach_team_games(stats: pd.DataFrame) -> pd.DataFrame:
    """Add `team_games`: games his (final) team played that season.

    The 2011-12, 2019-20 and 2020-21 seasons were shortened and 2019-20 teams played
    unequal numbers of games, so availability must be measured against the team's
    schedule, not a fixed 82. Falls back to the season median if the team is missing.
    """
    t = load_team_records()
    t = t.assign(team_games=t["w"] + t["l"])
    typical = t.groupby("season")["team_games"].median().rename("typical_games")
    out = stats.merge(t[["season", "team", "team_games"]], on=["season", "team"],
                      how="left", validate="many_to_one")
    out = out.join(typical, on="season")
    out["team_games"] = out["team_games"].fillna(out["typical_games"])
    return out.drop(columns="typical_games")


def load_salaries() -> pd.DataFrame:
    """Salaries 2010-2026 as (name_key, season, team, salary)."""
    hist = pd.read_csv(RAW_DIR / "salaries_2010_2025.csv")
    hist = hist.rename(columns={"Player": "player", "Year": "season",
                                "Team": "sal_team", "Salary": "salary"})
    cur = pd.read_csv(RAW_DIR / "salaries_2026.csv").rename(columns={"team": "sal_team"})
    cur["season"] = 2026
    sal = pd.concat([hist[["player", "season", "sal_team", "salary"]],
                     cur[["player", "season", "sal_team", "salary"]]], ignore_index=True)
    sal["name_key"] = sal["player"].map(name_key)
    sal = sal.dropna(subset=["salary"])
    # Same player listed twice in one season with identical salary (trade dupes)
    sal = sal.drop_duplicates(["name_key", "season", "salary"])
    return sal


def attach_salaries(stats: pd.DataFrame, sal: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join salaries to stats on (name_key, season), using team to break ties.

    Returns the stats frame with `salary` and `pct_cap`, plus a diagnostics
    dict that reports match rates and ambiguous cases.
    """
    seasons = sorted(sal["season"].unique())
    st = stats[stats["season"].isin(seasons)]
    cand = st[["player_id", "season", "name_key", "team"]].merge(
        sal[["name_key", "season", "sal_team", "salary"]], on=["name_key", "season"])

    # Where a name maps to >1 candidate on either side, require a team match.
    cand["n_stat"] = cand.groupby(["name_key", "season", "sal_team"])["player_id"].transform("nunique")
    cand["n_sal"] = cand.groupby(["player_id", "season"])["salary"].transform("size")
    ambiguous = (cand["n_stat"] > 1) | (cand["n_sal"] > 1)
    resolved = cand[~ambiguous | (cand["sal_team"] == cand["team"])]
    resolved = resolved.drop_duplicates(["player_id", "season"], keep=False)

    out = stats.merge(resolved[["player_id", "season", "salary"]],
                      on=["player_id", "season"], how="left", validate="one_to_one")
    out["salary_cap"] = out["season"].map(SALARY_CAP)
    out["pct_cap"] = out["salary"] / out["salary_cap"]

    # Diagnostics: share of meaningful-minute players without a salary match
    in_scope = out[out["season"].isin(seasons) & (out["mp"] >= SETTINGS.min_minutes_value)]
    unmatched_sal = sal.merge(resolved[["player_id", "season"]].merge(
        stats[["player_id", "season", "name_key"]]), on=["name_key", "season"],
        how="left", indicator=True)
    diag = {
        "match_rate_rotation": float(in_scope["salary"].notna().mean()),
        "n_ambiguous_dropped": int(ambiguous.sum() - (ambiguous & (cand["sal_team"] == cand["team"])).sum()),
        "unmatched_salary_rows": unmatched_sal.loc[unmatched_sal["_merge"] == "left_only",
                                                   ["player", "season", "salary"]],
        "unmatched_players_rotation": in_scope.loc[in_scope["salary"].isna(),
                                                 ["player", "season", "team", "mp"]],
    }
    log.info("Salary match rate (rotation players): %.1f%%", 100 * diag["match_rate_rotation"])
    return out, diag
