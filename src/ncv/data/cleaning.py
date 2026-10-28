"""Cleaning utilities shared by the loaders."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")
# Verified spelling differences between Basketball-Reference and the salary
# sources, as normalized keys (BBRef form -> salary-source form). Every entry
# was confirmed by auditing unmatched rows; add more only the same way.
NAME_ALIASES: dict[str, str] = {
    "mo bamba": "mohamed bamba",
    "gg jackson": "gregory jackson",
    "hugo gonzalez": "hugo gonzalez pena",
}


def name_key(name: object) -> str:
    """Normalize a player name for cross-source joins.

    Transliterates accents, lowercases, strips periods/apostrophes and
    generational suffixes, collapses whitespace, then applies known aliases.
    'Nikola Jokić' -> 'nikola jokic';  "Jaren Jackson Jr." -> 'jaren jackson'
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = _SUFFIX.sub("", s).strip()
    return NAME_ALIASES.get(s, s)


def collapse_multi_team(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season.

    Basketball-Reference lists traded players with a season-total row
    (team '2TM', '3TM', ...; 'TOT' in older exports) followed by one row per
    stint in chronological order. Keep the total row's stats and label it
    with the player's final team of the season.
    """
    df = df.copy()
    df["_order"] = range(len(df))
    is_total = df["team"].astype(str).str.fullmatch(r"\dTM|TOT")
    keys = ["player_id", "season"]

    last_team = (
        df[~is_total]
        .sort_values("_order")
        .groupby(keys)["team"]
        .last()
        .rename("final_team")
    )
    n_teams = df[~is_total].groupby(keys).size().rename("n_teams")

    has_total = df.assign(_t=is_total).groupby(keys)["_t"].transform("any")
    kept = df[(is_total) | (~has_total)].drop_duplicates(keys, keep="first")
    kept = kept.join(last_team, on=keys).join(n_teams, on=keys)
    kept["team"] = kept["final_team"].fillna(kept["team"])
    return kept.drop(columns=["_order", "final_team"]).reset_index(drop=True)
