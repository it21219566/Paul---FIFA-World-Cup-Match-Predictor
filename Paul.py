import os
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from scipy.stats import poisson
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
CACHE_DIR = "data_cache"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIXTURES_PATH = os.path.join(CACHE_DIR, "fixtures.csv")

ELO_BASE = 1500.0
ELO_K = 32
ELO_HOME_BONUS = 60

FEATURES = [
    "neutral", "tournament_weight", "home_elo", "away_elo", "elo_diff",
    "home_gf5", "home_ga5", "away_gf5", "away_ga5",
    "home_rest_days", "away_rest_days"
]


def fetch_results():
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "results.csv")
    if not os.path.exists(path):
        print("Downloading historical dataset (~3MB)...")
        resp = requests.get(RESULTS_URL, timeout=120)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
    return pd.read_csv(path)


# ==========================================
# 2. Feature Engineering Core
# ==========================================
def tournament_weight(name):
    t = str(name).lower()
    if "fifa world cup" in t and "qualif" not in t: return 4
    if "qualif" in t: return 3
    big = ["uefa nations", "copa america", "afc asian cup", "africa cup", "concacaf", "uefa euro"]
    if any(tok in t for tok in big): return 3
    if "friendly" in t: return 1
    return 2


def compute_elo_and_weights(r):
    print("Calculating historical Elo ratings...")
    r = r.sort_values("date").reset_index(drop=True)
    r["tournament_weight"] = r["tournament"].map(tournament_weight)

    rating, home_pre, away_pre = {}, np.zeros(len(r)), np.zeros(len(r))
    for i, row in r.iterrows():
        rh = rating.get(row.home_team, ELO_BASE)
        ra = rating.get(row.away_team, ELO_BASE)
        home_pre[i], away_pre[i] = rh, ra

        bonus = 0 if row.neutral == 1 else ELO_HOME_BONUS
        exp_home = 1 / (1 + 10 ** (-((rh + bonus) - ra) / 400))

        if row.home_score > row.away_score:
            score_home = 1.0
        elif row.home_score == row.away_score:
            score_home = 0.5
        else:
            score_home = 0.0

        margin = abs(int(row.home_score) - int(row.away_score))
        mult = np.log(max(margin, 1) + 1) * (2.2 / (abs(rh - ra) * 0.001 + 2.2))

        rating[row.home_team] = rh + ELO_K * mult * (score_home - exp_home)
        rating[row.away_team] = ra + ELO_K * mult * ((1 - score_home) - (1 - exp_home))

    r["home_elo"], r["away_elo"] = home_pre, away_pre
    r["elo_diff"] = home_pre - away_pre
    return r, rating


def add_form_features(r):
    print("Calculating rolling team forms and rest days...")
    home = pd.DataFrame({"date": r["date"], "team": r["home_team"], "gf": r["home_score"], "ga": r["away_score"]})
    away = pd.DataFrame({"date": r["date"], "team": r["away_team"], "gf": r["away_score"], "ga": r["home_score"]})
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)

    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["gf_lag"] = long.groupby("team")["gf"].shift(1)
    long["ga_lag"] = long.groupby("team")["ga"].shift(1)

    long["gf5"] = long.groupby("team")["gf_lag"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["ga5"] = long.groupby("team")["ga_lag"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days.fillna(30)

    form = long[["date", "team", "gf5", "ga5", "rest_days"]].drop_duplicates(["date", "team"])

    r = r.merge(
        form.rename(columns={"team": "home_team", "gf5": "home_gf5", "ga5": "home_ga5", "rest_days": "home_rest_days"}),
        on=["date", "home_team"], how="left")
    r = r.merge(
        form.rename(columns={"team": "away_team", "gf5": "away_gf5", "ga5": "away_ga5", "rest_days": "away_rest_days"}),
        on=["date", "away_team"], how="left")
    return r


# ==========================================
# 3. Startup Model Training
# ==========================================
print("Booting up ML Engine...")
raw_df = fetch_results()
raw_df['date'] = pd.to_datetime(raw_df['date'])
df_clean = raw_df.dropna(subset=['home_score', 'away_score']).copy()
df_clean['neutral'] = df_clean['neutral'].astype(str).str.upper().eq("TRUE").astype(int)

df_engineered, final_elo = compute_elo_and_weights(df_clean)
df_engineered = add_form_features(df_engineered)

train_df = df_engineered[df_engineered['date'] > '2000-01-01'].dropna(subset=FEATURES)

X = train_df[FEATURES]
y_home = train_df['home_score'].astype(int)
y_away = train_df['away_score'].astype(int)

print("Training Advanced Poisson XGBoost models...")
model_home = xgb.XGBRegressor(objective='count:poisson', n_estimators=150, max_depth=4, learning_rate=0.05)
model_away = xgb.XGBRegressor(objective='count:poisson', n_estimators=150, max_depth=4, learning_rate=0.05)
model_home.fit(X, y_home)
model_away.fit(X, y_away)
print("Models trained successfully. API is ready to serve!")

# ==========================================
# 4. Helper Functions & Fixtures
# ==========================================
valid_teams = set(raw_df['home_team']).union(set(raw_df['away_team']))
valid_teams_lower = {team.lower(): team for team in valid_teams}
NAME_ALIASES = {
    "usa": "United States", "us": "United States",
    "korea republic": "South Korea", "republic of ireland": "Ireland",
    "turkiye": "Turkey", "türkiye": "Turkey", "cape verde": "Cabo Verde",
    "cote d'ivoire": "Ivory Coast", "ivory coast": "Ivory Coast",
    "czechia": "Czech Republic", "curacao": "Curacao",
    "congo dr": "DR Congo", "dr congo": "DR Congo", "drc": "DR Congo",
    "congo": "Republic of the Congo"
}


def resolve_team(user_input):
    val = user_input.lower().strip()
    if val in NAME_ALIASES: return NAME_ALIASES[val]
    if val in valid_teams_lower: return valid_teams_lower[val]
    return None


FIXTURE_NAME_MAP = {
    "IR Iran": "Iran", "Korea Republic": "South Korea", "Türkiye": "Turkey",
    "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao", "USA": "United States",
    "Cape Verde": "Cabo Verde",
}


def map_fixture_name(name): return FIXTURE_NAME_MAP.get(name.strip(), name.strip())


def _side_matches(user_input, raw_name):
    u = user_input.strip().lower()
    return u in {raw_name.strip().lower(), map_fixture_name(raw_name).strip().lower()}


def find_fixture(team_a, team_b):
    if not os.path.exists(FIXTURES_PATH): return None
    try:
        fx = pd.read_csv(FIXTURES_PATH)
        for _, row in fx.iterrows():
            if " v " not in str(row.get("teams", "")): continue
            left, right = [p.strip() for p in str(row["teams"]).split(" v ")]
            if (_side_matches(team_a, left) and _side_matches(team_b, right)) or \
                    (_side_matches(team_a, right) and _side_matches(team_b, left)):
                return {
                    "group": row.get("group", "Unknown Group"),
                    "stadium": row.get("stadium", "Unknown Stadium"),
                    "date": row.get("date_dt", row.get("date", "Unknown Date"))
                }
    except Exception:
        pass
    return None


def get_current_stats(team_name, match_date="2026-06-11"):
    team_games = df_clean[(df_clean['home_team'] == team_name) | (df_clean['away_team'] == team_name)].sort_values(
        'date')
    if team_games.empty: return None
    last_5 = team_games.tail(5)
    gf_total, ga_total = 0, 0
    for _, row in last_5.iterrows():
        if row['home_team'] == team_name:
            gf_total += row['home_score']
            ga_total += row['away_score']
        else:
            gf_total += row['away_score']
            ga_total += row['home_score']
    last_match_date = pd.to_datetime(last_5.iloc[-1]['date'])
    match_date_dt = pd.to_datetime(match_date)
    rest_days = (match_date_dt - last_match_date).days
    return {
        "elo": final_elo.get(team_name, ELO_BASE),
        "gf5": gf_total / len(last_5),
        "ga5": ga_total / len(last_5),
        "rest": min(rest_days, 30)
    }


def predict_symmetric_poisson(team_A_stats, team_B_stats, is_neutral=1, weight=4):
    row_AB = pd.DataFrame([{
        "neutral": is_neutral, "tournament_weight": weight, "home_elo": team_A_stats['elo'],
        "away_elo": team_B_stats['elo'],
        "elo_diff": team_A_stats['elo'] - team_B_stats['elo'], "home_gf5": team_A_stats['gf5'],
        "home_ga5": team_A_stats['ga5'],
        "away_gf5": team_B_stats['gf5'], "away_ga5": team_B_stats['ga5'], "home_rest_days": team_A_stats['rest'],
        "away_rest_days": team_B_stats['rest']
    }])[FEATURES]
    row_BA = pd.DataFrame([{
        "neutral": is_neutral, "tournament_weight": weight, "home_elo": team_B_stats['elo'],
        "away_elo": team_A_stats['elo'],
        "elo_diff": team_B_stats['elo'] - team_A_stats['elo'], "home_gf5": team_B_stats['gf5'],
        "home_ga5": team_B_stats['ga5'],
        "away_gf5": team_A_stats['gf5'], "away_ga5": team_A_stats['ga5'], "home_rest_days": team_B_stats['rest'],
        "away_rest_days": team_A_stats['rest']
    }])[FEATURES]
    final_lambda_A = (model_home.predict(row_AB)[0] + model_away.predict(row_BA)[0]) / 2
    final_lambda_B = (model_away.predict(row_AB)[0] + model_home.predict(row_BA)[0]) / 2
    return float(final_lambda_A), float(final_lambda_B)


def generate_score_matrix(lambda_a, lambda_b, max_goals=10):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            matrix[x, y] = poisson.pmf(x, lambda_a) * poisson.pmf(y, lambda_b)
    return matrix


# ==========================================
# 5. FastAPI Application Initialization
# ==========================================
app = FastAPI(title="World Cup Predictor API")

# --- UI STARTUP LINK ---
# Gets the absolute path of index.html relative to this Python script
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, "index.html").replace("\\", "/")

print("\n" + "=" * 65)
print("🚀 BACKEND READY! The AI is listening on port 8000.")
print("🌐 CTRL+CLICK THE LINK BELOW TO OPEN THE FRONTEND UI:")
print(f"👉  file:///{frontend_path}")
print("=" * 65 + "\n")

# Enable CORS so the HTML frontend can talk to the Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/predict")
def predict_match(team1: str, team2: str):
    """API Endpoint to fetch a prediction for any two teams"""
    t1_resolved = resolve_team(team1)
    t2_resolved = resolve_team(team2)

    if not t1_resolved: raise HTTPException(status_code=404, detail=f"Team '{team1}' not found in database.")
    if not t2_resolved: raise HTTPException(status_code=404, detail=f"Team '{team2}' not found in database.")
    if t1_resolved == t2_resolved: raise HTTPException(status_code=400, detail="A team cannot play itself.")

    stats_1 = get_current_stats(t1_resolved)
    stats_2 = get_current_stats(t2_resolved)

    m = find_fixture(t1_resolved, t2_resolved)
    match_info = f"{m['date']} · {m['group']} · {m['stadium']}" if m else "Hypothetical / Friendly Match"

    l_1, l_2 = predict_symmetric_poisson(stats_1, stats_2, is_neutral=1, weight=4)
    matrix = generate_score_matrix(l_1, l_2)

    t1_win = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    t2_win = float(np.sum(np.triu(matrix, 1)))

    flat_probs = []
    for i in range(11):
        for j in range(11):
            flat_probs.append({"score1": i, "score2": j, "prob": float(matrix[i, j])})
    flat_probs.sort(key=lambda x: x["prob"], reverse=True)

    return {
        "match": {
            "team1": t1_resolved,
            "team2": t2_resolved,
            "info": match_info
        },
        "stats": {
            "team1": {"elo": round(stats_1['elo']), "gf": round(stats_1['gf5'], 2), "ga": round(stats_1['ga5'], 2)},
            "team2": {"elo": round(stats_2['elo']), "gf": round(stats_2['gf5'], 2), "ga": round(stats_2['ga5'], 2)}
        },
        "xg": {
            "team1": round(l_1, 2),
            "team2": round(l_2, 2)
        },
        "probabilities": {
            "team1": round(t1_win * 100, 1),
            "draw": round(draw * 100, 1),
            "team2": round(t2_win * 100, 1)
        },
        "top_scores": [
            {"score1": s["score1"], "score2": s["score2"], "prob": round(s["prob"] * 100, 1)}
            for s in flat_probs[:3]
        ]
    }