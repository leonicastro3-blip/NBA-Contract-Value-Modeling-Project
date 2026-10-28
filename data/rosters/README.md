# Roster and contract overrides

Current rosters and 2026-27 salaries come from a Spotrac snapshot (captured
2026-10-09) that `scripts/download_data.py` fetches. Use these files only to
override it, e.g. after a trade:

- `<TEAM>_2027.csv` — column `player`. If present, the dashboard offers a toggle
  to use this list instead of the snapshot roster for that team.
- `contracts_2027.csv` — columns `player,salary`. Overrides individual salaries.
