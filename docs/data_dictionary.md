# Data Dictionary

Schemas for the committed sample datasets in `data/sample/`.

## sample_games.csv

| Column | Description |
| --- | --- |
| game_id | Unique game identifier (e.g. `SAMPLE_GAME_001`). |
| season | NBA season label (e.g. `2024-25`). |
| game_date | Date the game was played (YYYY-MM-DD). |
| home_team | Home team name. |
| away_team | Away team name. |
| home_team_id | Numeric home team identifier. |
| away_team_id | Numeric away team identifier. |
| status | Game status (e.g. `Final`). |
| game_type | `regular` or `playoff`. |

## sample_pregame_features.csv

| Column | Description |
| --- | --- |
| game_id | Game identifier. |
| season | Season label. |
| game_date | Date of game. |
| home_team / away_team | Team names. |
| home_win_pct / away_win_pct | Season win percentage entering the game. |
| win_pct_diff | `home_win_pct - away_win_pct`. |
| home_net_rating / away_net_rating | Net rating (points per 100 possessions). |
| net_rating_diff | `home_net_rating - away_net_rating`. |
| home_rest_days / away_rest_days | Days of rest before the game. |
| rest_days_diff | `home_rest_days - away_rest_days`. |
| home_recent_form / away_recent_form | Recent win rate (last N games), 0-1. |
| recent_form_diff | `home_recent_form - away_recent_form`. |
| is_playoff | 1 if playoff game, else 0. |
| home_team_won | Target label: 1 if home team won, else 0. |

## sample_live_features.csv

| Column | Description |
| --- | --- |
| game_id | Game identifier. |
| event_num | Sequential event index within the game. |
| period | Quarter/period number. |
| seconds_remaining_period | Seconds left in the current period. |
| seconds_remaining_game | Seconds left in the whole game. |
| home_score / away_score | Running score at the event. |
| score_margin_home | `home_score - away_score`. |
| abs_score_margin | Absolute score margin. |
| event_type | Type of play-by-play event. |
| home_team_won | Target label: 1 if home team won, else 0. |

## sample_predictions.csv

| Column | Description |
| --- | --- |
| game_id | Game identifier. |
| event_num | Sequential event index. |
| period | Quarter/period number. |
| seconds_remaining_game | Seconds left in the whole game. |
| home_team / away_team | Team names. |
| home_score / away_score | Running score. |
| home_win_probability | Predicted P(home win), 0-1. |
| away_win_probability | Predicted P(away win), 0-1. |
| predicted_winner | Team name with the higher probability. |

## data/manual/postgame_results.csv (written by the dashboard)

| Column | Description |
| --- | --- |
| game_id | Game identifier. |
| home_score / away_score | Final score entered by hand. |
| winner | `home`, `away`, or `tie` derived from the scores. |
| source | `manual` or `corrected_manual`. |
| confirmed_at | UTC timestamp when recorded. |
