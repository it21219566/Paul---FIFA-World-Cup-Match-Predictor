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
GOALSCORERS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
FIXTURES_PATH = os.path.join(CACHE_DIR, "fixtures.csv")

ELO_BASE = 1500.0
ELO_K = 32
ELO_HOME_BONUS = 60

# The expanded Professional Feature Set
FEATURES = [
    "neutral", "tournament_weight", "home_elo", "away_elo", "elo_diff",
    "home_gf5", "home_ga5", "home_npg5", "away_gf5", "away_ga5", "away_npg5",
    "home_rest_days", "away_rest_days",
    "home_win5", "home_win10", "away_win5", "away_win10",
    "h2h_n", "h2h_home_winrate", "h2h_home_gd"
]


def fetch_data():
    os.makedirs(CACHE_DIR, exist_ok=True)
    r_path = os.path.join(CACHE_DIR, "results.csv")
    g_path = os.path.join(CACHE_DIR, "goalscorers.csv")

    if not os.path.exists(r_path):
        print("Downloading match results dataset (~3MB)...")
        open(r_path, "wb").write(requests.get(RESULTS_URL, timeout=120).content)
    if not os.path.exists(g_path):
        print("Downloading goalscorers dataset (~2MB)...")
        open(g_path, "wb").write(requests.get(GOALSCORERS_URL, timeout=120).content)

    return pd.read_csv(r_path), pd.read_csv(g_path)


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


def merge_goalscorers(results_df, scorers_df):
    print("Fusing open-play (non-penalty) goal data...")
    results_df['date'] = pd.to_datetime(results_df['date'])
    scorers_df['date'] = pd.to_datetime(scorers_df['date'])

    # Filter out penalties and own goals to find pure attacking strength
    scorers_df['penalty'] = scorers_df['penalty'].astype(str).str.lower().isin(['true', '1', 't'])
    scorers_df['own_goal'] = scorers_df['own_goal'].astype(str).str.lower().isin(['true', '1', 't'])
    open_play = scorers_df[(~scorers_df['penalty']) & (~scorers_df['own_goal'])]

    # Count non-penalty goals (npg) per team per match
    npg_counts = open_play.groupby(['date', 'home_team', 'away_team', 'team']).size().reset_index(name='npg')

    npg_dict = {(row['date'], row['home_team'], row['away_team'], row['team']): row['npg'] for _, row in
                npg_counts.iterrows()}

    home_npg, away_npg = [], []
    for _, row in results_df.iterrows():
        d, h, a = row['date'], row['home_team'], row['away_team']
        h_val = npg_dict.get((d, h, a, h), None)
        a_val = npg_dict.get((d, h, a, a), None)

        # Fallback to official score if scorer data is missing (historical matches)
        home_npg.append(h_val if h_val is not None else row['home_score'])
        away_npg.append(a_val if a_val is not None else row['away_score'])

    results_df['home_npg'] = home_npg
    results_df['away_npg'] = away_npg
    return results_df


def compute_elo_and_weights(r):
    print("Calculating dynamic historical Elo ratings...")
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


def add_advanced_features(r):
    print("Calculating 10-match forms, win rates, and Head-to-Head records...")
    home = pd.DataFrame(
        {"date": r["date"], "team": r["home_team"], "opp": r["away_team"], "gf": r["home_score"], "ga": r["away_score"],
         "npg": r["home_npg"]})
    away = pd.DataFrame(
        {"date": r["date"], "team": r["away_team"], "opp": r["home_team"], "gf": r["away_score"], "ga": r["home_score"],
         "npg": r["away_npg"]})
    long = pd.concat([home, away], ignore_index=True)

    long["result"] = np.where(long["gf"] > long["ga"], 1.0, np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["gd"] = long["gf"] - long["ga"]

    # 1. Team Form
    long_form = long.sort_values(["team", "date"]).reset_index(drop=True)
    long_form["prev_date"] = long_form.groupby("team")["date"].shift(1)
    long_form["result_lag"] = long_form.groupby("team")["result"].shift(1)
    long_form["gf_lag"] = long_form.groupby("team")["gf"].shift(1)
    long_form["ga_lag"] = long_form.groupby("team")["ga"].shift(1)
    long_form["npg_lag"] = long_form.groupby("team")["npg"].shift(1)

    long_form["win5"] = long_form.groupby("team")["result_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()).fillna(0.5)
    long_form["win10"] = long_form.groupby("team")["result_lag"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()).fillna(0.5)
    long_form["gf5"] = long_form.groupby("team")["gf_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()).fillna(1.0)
    long_form["ga5"] = long_form.groupby("team")["ga_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()).fillna(1.0)
    long_form["npg5"] = long_form.groupby("team")["npg_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()).fillna(1.0)
    long_form["rest_days"] = (long_form["date"] - long_form["prev_date"]).dt.days.fillna(30)

    form = long_form[["date", "team", "win5", "win10", "gf5", "ga5", "npg5", "rest_days"]].drop_duplicates(
        ["date", "team"])
    r = r.merge(form.rename(
        columns={"team": "home_team", "win5": "home_win5", "win10": "home_win10", "gf5": "home_gf5", "ga5": "home_ga5",
                 "npg5": "home_npg5", "rest_days": "home_rest_days"}), on=["date", "home_team"], how="left")
    r = r.merge(form.rename(
        columns={"team": "away_team", "win5": "away_win5", "win10": "away_win10", "gf5": "away_gf5", "ga5": "away_ga5",
                 "npg5": "away_npg5", "rest_days": "away_rest_days"}), on=["date", "away_team"], how="left")

    # 2. Head-to-Head (H2H)
    long_h2h = long.sort_values(["team", "opp", "date"]).reset_index(drop=True)
    g = long_h2h.groupby(["team", "opp"])
    long_h2h["h2h_n"] = g.cumcount()
    long_h2h["h2h_winrate"] = g["result"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean()).fillna(0.5)
    long_h2h["h2h_gd"] = g["gd"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean()).fillna(0.0)

    h2h = long_h2h[["date", "team", "opp", "h2h_n", "h2h_winrate", "h2h_gd"]].drop_duplicates(["date", "team", "opp"])
    r = r.merge(h2h.rename(
        columns={"team": "home_team", "opp": "away_team", "h2h_winrate": "h2h_home_winrate", "h2h_gd": "h2h_home_gd"}),
                on=["date", "home_team", "away_team"], how="left")

    return r


# ==========================================
# 3. Startup Model Training
# ==========================================
print("Booting up ML Engine...")
raw_df, scorers_df = fetch_data()
raw_df = raw_df.dropna(subset=['home_score', 'away_score']).copy()
raw_df['neutral'] = raw_df['neutral'].astype(str).str.upper().eq("TRUE").astype(int)

df_clean = merge_goalscorers(raw_df, scorers_df)
df_engineered, final_elo = compute_elo_and_weights(df_clean)
df_engineered = add_advanced_features(df_engineered)

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
# 4. Extraction & Inference Helpers
# ==========================================
valid_teams = set(raw_df['home_team']).union(set(raw_df['away_team']))
valid_teams_lower = {team.lower(): team for team in valid_teams}
NAME_ALIASES = {
    "usa": "United States", "us": "United States", "korea republic": "South Korea",
    "republic of ireland": "Ireland", "turkiye": "Turkey", "türkiye": "Turkey",
    "cape verde": "Cabo Verde", "cote d'ivoire": "Ivory Coast", "ivory coast": "Ivory Coast",
    "czechia": "Czech Republic", "curacao": "Curacao", "congo dr": "DR Congo",
    "dr congo": "DR Congo", "drc": "DR Congo", "congo": "Republic of the Congo"
}


def resolve_team(user_input):
    val = user_input.lower().strip()
    if val in NAME_ALIASES: return NAME_ALIASES[val]
    if val in valid_teams_lower: return valid_teams_lower[val]
    return None


def find_fixture(team_a, team_b):
    if not os.path.exists(FIXTURES_PATH): return None
    try:
        fx = pd.read_csv(FIXTURES_PATH)
        for _, row in fx.iterrows():
            if " v " not in str(row.get("teams", "")): continue
            left, right = [p.strip().lower() for p in str(row["teams"]).split(" v ")]
            a, b = team_a.lower(), team_b.lower()
            if (a in left and b in right) or (a in right and b in left):
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

    last_10 = team_games.tail(10)
    last_5 = team_games.tail(5)

    gf5, ga5, npg5, win5, win10 = 0, 0, 0, 0, 0

    for _, row in last_10.iterrows():
        is_h = row['home_team'] == team_name
        gf = row['home_score'] if is_h else row['away_score']
        ga = row['away_score'] if is_h else row['home_score']
        win10 += 1 if gf > ga else (0.5 if gf == ga else 0)

    for _, row in last_5.iterrows():
        is_h = row['home_team'] == team_name
        gf = row['home_score'] if is_h else row['away_score']
        ga = row['away_score'] if is_h else row['home_score']
        npg = row['home_npg'] if is_h else row['away_npg']

        gf5 += gf;
        ga5 += ga;
        npg5 += npg
        win5 += 1 if gf > ga else (0.5 if gf == ga else 0)

    last_match_date = pd.to_datetime(last_5.iloc[-1]['date'])
    rest_days = min((pd.to_datetime(match_date) - last_match_date).days, 30)

    return {
        "elo": final_elo.get(team_name, ELO_BASE),
        "gf5": gf5 / len(last_5), "ga5": ga5 / len(last_5), "npg5": npg5 / len(last_5),
        "win5": win5 / len(last_5), "win10": win10 / len(last_10), "rest": rest_days
    }


def get_h2h_stats(teamA, teamB):
    h2h_games = df_clean[((df_clean['home_team'] == teamA) & (df_clean['away_team'] == teamB)) |
                         ((df_clean['home_team'] == teamB) & (df_clean['away_team'] == teamA))]
    n = len(h2h_games)
    if n == 0: return {"n": 0, "winrate": 0.5, "gd": 0.0}

    wins_A, gd_A = 0, 0
    for _, row in h2h_games.iterrows():
        if row['home_team'] == teamA:
            gd_A += (row['home_score'] - row['away_score'])
            wins_A += 1 if row['home_score'] > row['away_score'] else (
                0.5 if row['home_score'] == row['away_score'] else 0)
        else:
            gd_A += (row['away_score'] - row['home_score'])
            wins_A += 1 if row['away_score'] > row['home_score'] else (
                0.5 if row['away_score'] == row['home_score'] else 0)

    return {"n": n, "winrate": wins_A / n, "gd": gd_A / n}


def predict_symmetric_poisson(stats_A, stats_B, h2h, is_neutral=1, weight=4):
    row_AB = pd.DataFrame([{
        "neutral": is_neutral, "tournament_weight": weight,
        "home_elo": stats_A['elo'], "away_elo": stats_B['elo'], "elo_diff": stats_A['elo'] - stats_B['elo'],
        "home_gf5": stats_A['gf5'], "home_ga5": stats_A['ga5'], "home_npg5": stats_A['npg5'],
        "away_gf5": stats_B['gf5'], "away_ga5": stats_B['ga5'], "away_npg5": stats_B['npg5'],
        "home_rest_days": stats_A['rest'], "away_rest_days": stats_B['rest'],
        "home_win5": stats_A['win5'], "home_win10": stats_A['win10'],
        "away_win5": stats_B['win5'], "away_win10": stats_B['win10'],
        "h2h_n": h2h['n'], "h2h_home_winrate": h2h['winrate'], "h2h_home_gd": h2h['gd']
    }])[FEATURES]

    row_BA = pd.DataFrame([{
        "neutral": is_neutral, "tournament_weight": weight,
        "home_elo": stats_B['elo'], "away_elo": stats_A['elo'], "elo_diff": stats_B['elo'] - stats_A['elo'],
        "home_gf5": stats_B['gf5'], "home_ga5": stats_B['ga5'], "home_npg5": stats_B['npg5'],
        "away_gf5": stats_A['gf5'], "away_ga5": stats_A['ga5'], "away_npg5": stats_A['npg5'],
        "home_rest_days": stats_B['rest'], "away_rest_days": stats_A['rest'],
        "home_win5": stats_B['win5'], "home_win10": stats_B['win10'],
        "away_win5": stats_A['win5'], "away_win10": stats_A['win10'],
        "h2h_n": h2h['n'], "h2h_home_winrate": 1.0 - h2h['winrate'], "h2h_home_gd": -h2h['gd']  # Flipped perspective
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

current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, "index.html").replace("\\", "/")

print("\n" + "=" * 65)
print("🚀 BACKEND READY! The AI is listening on port 8000.")
print("🌐 CTRL+CLICK THE LINK BELOW TO OPEN THE FRONTEND UI:")
print(f"👉  file:///{frontend_path}")
print("=" * 65 + "\n")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/predict")
def predict_match(team1: str, team2: str):
    t1_resolved = resolve_team(team1)
    t2_resolved = resolve_team(team2)

    if not t1_resolved: raise HTTPException(status_code=404, detail=f"Team '{team1}' not found in database.")
    if not t2_resolved: raise HTTPException(status_code=404, detail=f"Team '{team2}' not found in database.")
    if t1_resolved == t2_resolved: raise HTTPException(status_code=400, detail="A team cannot play itself.")

    stats_1 = get_current_stats(t1_resolved)
    stats_2 = get_current_stats(t2_resolved)
    h2h_stats = get_h2h_stats(t1_resolved, t2_resolved)

    m = find_fixture(t1_resolved, t2_resolved)
    match_info = f"{m['date']} · {m['group']} · {m['stadium']}" if m else "Hypothetical / Friendly Match"

    l_1, l_2 = predict_symmetric_poisson(stats_1, stats_2, h2h_stats, is_neutral=1, weight=4)
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
        "match": {"team1": t1_resolved, "team2": t2_resolved, "info": match_info},
        "stats": {
            "team1": {"elo": round(stats_1['elo']), "gf": round(stats_1['gf5'], 2), "ga": round(stats_1['ga5'], 2)},
            "team2": {"elo": round(stats_2['elo']), "gf": round(stats_2['gf5'], 2), "ga": round(stats_2['ga5'], 2)}
        },
        "xg": {"team1": round(l_1, 2), "team2": round(l_2, 2)},
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