#!/usr/bin/env python3
"""
Premier League Football Match Outcome Predictor

Features:
- Downloads historical Premier League data from football-data.co.uk
- Engineers features including Elo ratings, home advantage, rolling form and goals stats
- Trains a multinomial Logistic Regression model with time-series cross-validation
- Evaluates with accuracy and log-loss (optionally Brier score)
- Persists model and team state to disk
- CLI to train/evaluate and predict matches: 'train' and 'predict --home TEAM --away TEAM'

Usage examples:
  python main.py train --start-season 2010 --end-season auto
  python main.py predict --home Arsenal --away "Manchester City"
  python main.py list-teams

Requirements (install once):
  pip install pandas numpy scikit-learn requests joblib

Note:
- Team names should match the dataset's canonical names listed by 'list-teams'.
- The script caches downloaded raw CSVs under data/raw/ for reproducibility.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ---------------------- Configuration ----------------------
LEAGUE_CODE = "E0"  # Premier League
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
STATE_PATH = os.path.join(MODELS_DIR, "state.joblib")

# Elo configuration
HOME_ELO_ADV = 65.0
ELO_K = 20.0
ROLLING_N = 5

np.random.seed(42)

# Common team name canonicalization (partial, robust to typical variants)
TEAM_ALIASES = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton",
    "Wolverhampton Wanderers": "Wolverhampton",
    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "West Brom": "West Bromwich Albion",
    "Sheffield Utd": "Sheffield United",
    "Newcastle Utd": "Newcastle",
    "Newcastle United": "Newcastle",
    "Cardiff": "Cardiff City",
    "Birmingham": "Birmingham City",
    "Hull": "Hull City",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "West Ham": "West Ham United",
    "Nott'm Forest": "Nottingham Forest",
    "Nott Forest": "Nottingham Forest",
    "QPR": "Queens Park Rangers",
    "Blackburn": "Blackburn Rovers",
    "Bolton": "Bolton Wanderers",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Middlesbrough": "Middlesbrough",
    "Derby": "Derby County",
    "Portsmouth": "Portsmouth",
}

# ---------------------- Data structures ----------------------
@dataclass
class TeamState:
    rating: float = 1500.0
    results: deque = field(default_factory=lambda: deque(maxlen=ROLLING_N))  # values: 1 win, 0.5 draw, 0 loss
    goals_for: deque = field(default_factory=lambda: deque(maxlen=ROLLING_N))
    goals_against: deque = field(default_factory=lambda: deque(maxlen=ROLLING_N))
    last_match_date: Optional[pd.Timestamp] = None

    def as_features(self) -> Dict[str, float]:
        # Form (wins as 1, draw as 0.5, loss as 0)
        if len(self.results) > 0:
            form = float(np.mean(self.results))
        else:
            form = 0.5  # neutral prior

        gf_avg = float(np.mean(self.goals_for)) if len(self.goals_for) > 0 else 1.3
        ga_avg = float(np.mean(self.goals_against)) if len(self.goals_against) > 0 else 1.3
        gd_avg = gf_avg - ga_avg
        matches = float(len(self.results))
        return {
            "rating": self.rating,
            "form": form,
            "gf_avg": gf_avg,
            "ga_avg": ga_avg,
            "gd_avg": gd_avg,
            "matches": matches,
        }

# ---------------------- Utilities ----------------------
def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def season_codes(start_season: int, end_season: Optional[int] = None) -> List[str]:
    """
    Build football-data season codes from start to end inclusive.
    start_season: the starting year of the season (e.g., 2010 for 2010/11)
    end_season: ending year (e.g., 2024 for 2024/25). If None or 'auto', goes to current.
    """
    if end_season is None:
        # Guess latest season code based on current year
        today = datetime.today()
        # If we're before August, current season probably spans prev/current
        if today.month < 8:
            end_season = today.year
        else:
            end_season = today.year
    codes = []
    for y in range(start_season, end_season + 1):
        a = str(y % 100).zfill(2)
        b = str((y + 1) % 100).zfill(2)
        codes.append(f"{a}{b}")
    return codes

def canonical_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)

# ---------------------- Data loading ----------------------
def download_season(season_code: str, league: str = LEAGUE_CODE) -> Optional[pd.DataFrame]:
    url = BASE_URL.format(season=season_code, league=league)
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200 or len(resp.content) == 0:
            return None
        bio = io.BytesIO(resp.content)
        df = pd.read_csv(bio)
        if df.empty:
            return None
        df["season_code"] = season_code
        return df
    except Exception:
        return None

EXPECTED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

DATE_COLS = ["Date"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Some seasons have different capitalization or extra spaces
    df = df.rename(columns={c: c.strip() for c in df.columns})
    mapping = {}
    for col in EXPECTED_COLS:
        if col in df.columns:
            mapping[col] = col
        else:
            # Known common variants
            variants = {
                "FTHG": ["FTHG"],
                "FTAG": ["FTAG"],
                "FTR": ["FTR"],
                "HomeTeam": ["HomeTeam"],
                "AwayTeam": ["AwayTeam"],
                "Date": ["Date"],
            }
            for v in variants[col]:
                if v in df.columns:
                    mapping[v] = col
                    break
    df = df.rename(columns=mapping)
    # Retain only expected columns if possible
    cols = [c for c in EXPECTED_COLS if c in df.columns]
    df = df[cols]
    # Parse dates robustly
    if "Date" in df.columns:
        # football-data often uses dayfirst format
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    # Cast goal columns to numeric
    for g in ["FTHG", "FTAG"]:
        if g in df.columns:
            df[g] = pd.to_numeric(df[g], errors="coerce")
    # Canonicalize team names
    for tcol in ["HomeTeam", "AwayTeam"]:
        if tcol in df.columns:
            df[tcol] = df[tcol].astype(str).map(canonical_team)
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])  # allow missing results to be dropped later


def load_data(start_season: int = 2010, end_season: Optional[int] = None, cache: bool = True) -> pd.DataFrame:
    ensure_dirs()
    codes = season_codes(start_season, end_season)
    frames = []
    for code in codes:
        cached_path = os.path.join(RAW_DIR, f"{code}_{LEAGUE_CODE}.csv")
        df = None
        if cache and os.path.exists(cached_path):
            try:
                df = pd.read_csv(cached_path)
            except Exception:
                df = None
        if df is None:
            df = download_season(code)
            if df is not None and cache:
                try:
                    df.to_csv(cached_path, index=False)
                except Exception:
                    pass
        if df is None or df.empty:
            continue
        df = normalize_columns(df)
        df["Season"] = code
        frames.append(df)
    if not frames:
        raise RuntimeError("No data downloaded/loaded. Try widening the season range or check network connection.")
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    # Filter to rows with result available
    all_df = all_df.dropna(subset=["FTR", "FTHG", "FTAG"])  # keep only finished matches
    # Keep only expected results
    all_df = all_df[all_df["FTR"].isin(["H", "D", "A"])].copy()
    # Sort chronologically
    all_df = all_df.sort_values("Date").reset_index(drop=True)
    return all_df

# ---------------------- Feature engineering ----------------------
def elo_expectation(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


def update_elo(home: TeamState, away: TeamState, result: str):
    # Home rating includes home field advantage
    home_effective = home.rating + HOME_ELO_ADV
    away_effective = away.rating
    e_home = elo_expectation(home_effective, away_effective)
    e_away = 1.0 - e_home

    if result == "H":
        s_home, s_away = 1.0, 0.0
    elif result == "A":
        s_home, s_away = 0.0, 1.0
    else:
        s_home, s_away = 0.5, 0.5

    home.rating += ELO_K * (s_home - e_home)
    away.rating += ELO_K * (s_away - e_away)


def update_rolling(home: TeamState, away: TeamState, hg: int, ag: int, result: str, date: pd.Timestamp):
    # From team's perspective results
    if result == "H":
        home_res, away_res = 1.0, 0.0
    elif result == "A":
        home_res, away_res = 0.0, 1.0
    else:
        home_res, away_res = 0.5, 0.5

    home.results.append(home_res)
    away.results.append(away_res)
    home.goals_for.append(float(hg))
    home.goals_against.append(float(ag))
    away.goals_for.append(float(ag))
    away.goals_against.append(float(hg))

    home.last_match_date = date
    away.last_match_date = date


def team_features(prefix: str, ts: TeamState) -> Dict[str, float]:
    base = ts.as_features()
    return {f"{prefix}_{k}": v for k, v in base.items()}


def match_features(home: TeamState, away: TeamState, match_date: Optional[pd.Timestamp]) -> Dict[str, float]:
    f = {}
    f.update(team_features("home", home))
    f.update(team_features("away", away))
    # Elo differential including home advantage
    f["elo_diff"] = (home.rating + HOME_ELO_ADV) - away.rating
    # Rest days (capped) if available
    def rest_days(ts: TeamState) -> float:
        if ts.last_match_date is not None and match_date is not None:
            return float(min(21, max(0, (match_date - ts.last_match_date).days)))
        return 7.0  # neutral prior
    f["home_rest_days"] = rest_days(home)
    f["away_rest_days"] = rest_days(away)
    # Cold start indicators
    f["home_cold"] = 1.0 if home.as_features()["matches"] < 1 else 0.0
    f["away_cold"] = 1.0 if away.as_features()["matches"] < 1 else 0.0
    return f


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, TeamState]]:
    states: Dict[str, TeamState] = defaultdict(TeamState)
    feature_rows: List[Dict[str, float]] = []
    labels: List[str] = []

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        date = row["Date"]
        hg = int(row["FTHG"]) if not pd.isna(row["FTHG"]) else None
        ag = int(row["FTAG"]) if not pd.isna(row["FTAG"]) else None
        res = row["FTR"]

        # Compute pre-match features from current state (no leakage)
        home_state = states[home_team]
        away_state = states[away_team]
        feats = match_features(home_state, away_state, date)
        feats["Season"] = row.get("Season", "")
        feats["Date_ordinal"] = pd.to_datetime(date).toordinal() if pd.notna(date) else np.nan
        feature_rows.append(feats)
        labels.append(res)

        # Update states post-match
        if hg is not None and ag is not None and res in {"H", "D", "A"}:
            update_elo(home_state, away_state, res)
            update_rolling(home_state, away_state, hg, ag, res, date)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="FTR")

    # Drop any rows with missing values (should be minimal due to priors)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]

    return X, y, states

# ---------------------- Modeling ----------------------
FEATURE_COLUMNS = [
    # Per-team features
    "home_rating", "home_form", "home_gf_avg", "home_ga_avg", "home_gd_avg", "home_matches",
    "away_rating", "away_form", "away_gf_avg", "away_ga_avg", "away_gd_avg", "away_matches",
    # Differential and situational
    "elo_diff", "home_rest_days", "away_rest_days", "home_cold", "away_cold",
    # Temporal feature
    "Date_ordinal",
]


def time_series_cv_scores(X: pd.DataFrame, y: pd.Series, n_splits: int = 6) -> Dict[str, float]:
    X_use = X[FEATURE_COLUMNS].values
    classes = np.array(["H", "D", "A"])  # fixed ordering

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, lls = [], []

    for train_idx, test_idx in tscv.split(X_use):
        X_tr, X_te = X_use[train_idx], X_use[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(multi_class="multinomial", max_iter=200, C=1.0, solver="lbfgs")),
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)
        pred = pipe.predict(X_te)

        accs.append(accuracy_score(y_te, pred))
        # Compute log loss with consistent class order
        # scikit ensures predict_proba columns align with classes_ attribute order
        lls.append(log_loss(y_te, proba, labels=pipe.named_steps["clf"].classes_))

    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_log_loss_mean": float(np.mean(lls)),
        "cv_log_loss_std": float(np.std(lls)),
    }


def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    X_use = X[FEATURE_COLUMNS].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", max_iter=500, C=1.0, solver="lbfgs")),
    ])
    pipe.fit(X_use, y)
    return pipe

# ---------------------- Prediction ----------------------

def fit_states_to_date(df: pd.DataFrame) -> Dict[str, TeamState]:
    # Walk through all matches to build latest states
    states: Dict[str, TeamState] = defaultdict(TeamState)
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    for _, row in df_sorted.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]
        if pd.isna(row["FTR"]) or pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]):
            continue
        res = row["FTR"]
        hg = int(row["FTHG"]) if not pd.isna(row["FTHG"]) else 0
        ag = int(row["FTAG"]) if not pd.isna(row["FTAG"]) else 0
        update_elo(states[home], states[away], res)
        update_rolling(states[home], states[away], hg, ag, res, date)
    return states


def features_for_match(home_team: str, away_team: str, states: Dict[str, TeamState], ref_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    home = states.get(home_team, TeamState())
    away = states.get(away_team, TeamState())
    feats = match_features(home, away, ref_date)
    feats["Date_ordinal"] = pd.to_datetime(ref_date if ref_date is not None else pd.Timestamp.today()).toordinal()
    X = pd.DataFrame([feats])
    # Ensure all expected columns present
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0.0
    return X[FEATURE_COLUMNS]


def predict_single(pipe: Pipeline, home: str, away: str, states: Dict[str, TeamState]) -> Tuple[np.ndarray, List[str]]:
    Xp = features_for_match(home, away, states)
    proba = pipe.predict_proba(Xp.values)[0]
    classes = list(pipe.named_steps["clf"].classes_)
    return proba, classes

# ---------------------- CLI orchestration ----------------------

def cmd_train(args):
    end_season = None if args.end_season == "auto" else int(args.end_season)
    df = load_data(start_season=int(args.start_season), end_season=end_season, cache=True)
    X, y, _ = build_dataset(df)
    print(f"Loaded {len(df)} matches; dataset rows: {len(X)}")

    scores = time_series_cv_scores(X, y, n_splits=int(args.cv_splits))
    print(json.dumps(scores, indent=2))

    model = train_model(X, y)
    ensure_dirs()
    joblib.dump(model, MODEL_PATH)
    # Save latest states for prediction convenience
    states = fit_states_to_date(df)
    joblib.dump(states, STATE_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Team states saved to {STATE_PATH}")


def cmd_predict(args):
    ensure_dirs()
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STATE_PATH):
        print("Model or state not found. Training first...")
        # Auto-train with defaults
        df = load_data(start_season=2010, end_season=None, cache=True)
        X, y, _ = build_dataset(df)
        model = train_model(X, y)
        joblib.dump(model, MODEL_PATH)
        states = fit_states_to_date(df)
        joblib.dump(states, STATE_PATH)
    else:
        model = joblib.load(MODEL_PATH)
        states = joblib.load(STATE_PATH)

    home = canonical_team(args.home)
    away = canonical_team(args.away)

    proba, classes = predict_single(model, home, away, states)
    mapping = dict(zip(classes, proba))
    # Present in H/D/A order
    out = {"Home": float(mapping.get("H", 0.0)), "Draw": float(mapping.get("D", 0.0)), "Away": float(mapping.get("A", 0.0))}

    print(json.dumps({
        "home_team": home,
        "away_team": away,
        "probabilities": out,
    }, indent=2))


def cmd_list_teams(args):
    df = load_data(start_season=int(args.start_season), end_season=None if args.end_season == "auto" else int(args.end_season), cache=True)
    teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
    print(json.dumps(teams, indent=2))


# ---------------------- Entry point ----------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Premier League Predictor")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Download data, train model, and save artifacts")
    p_train.add_argument("--start-season", default=2010, help="Start season start-year, e.g., 2010 for 2010/11")
    p_train.add_argument("--end-season", default="auto", help="End season start-year, e.g., 2024 for 2024/25, or 'auto'")
    p_train.add_argument("--cv-splits", default=6, help="Number of time-series CV splits")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Predict a single match outcome")
    p_pred.add_argument("--home", required=True, help="Home team name")
    p_pred.add_argument("--away", required=True, help="Away team name")
    p_pred.set_defaults(func=cmd_predict)

    p_list = sub.add_parser("list-teams", help="List known canonical team names from the dataset")
    p_list.add_argument("--start-season", default=2010, help="Start season start-year")
    p_list.add_argument("--end-season", default="auto", help="End season start-year or 'auto'")
    p_list.set_defaults(func=cmd_list_teams)

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
