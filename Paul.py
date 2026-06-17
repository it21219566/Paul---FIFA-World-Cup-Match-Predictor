import os
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from scipy.stats import poisson
import warnings

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
# 3. Model Training
# ==========================================
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

print("Training Advanced Poisson XGBoost models (this takes a few seconds)...")
model_home = xgb.XGBRegressor(objective='count:poisson', n_estimators=150, max_depth=4, learning_rate=0.05)
model_away = xgb.XGBRegressor(objective='count:poisson', n_estimators=150, max_depth=4, learning_rate=0.05)
model_home.fit(X, y_home)
model_away.fit(X, y_away)
print("Models trained successfully!\n")


# ==========================================
# 4. Real-World Stats Extractor & Predictor
# ==========================================
def get_current_stats(team_name, match_date="2026-06-11"):
    """Fetches the latest Elo and form up to the present day for a specific team."""
    # Use df_clean instead of raw_df to avoid grabbing future unplayed fixtures with NaN scores
    team_games = df_clean[(df_clean['home_team'] == team_name) | (df_clean['away_team'] == team_name)].sort_values(
        'date')
    if team_games.empty:
        return None

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
        "rest": min(rest_days, 30)  # Cap rest days at 30 so long offseasons don't break the model
    }


def predict_symmetric_poisson(team_A_stats, team_B_stats, is_neutral=1, weight=4):
    row_AB = pd.DataFrame([{
        "neutral": is_neutral, "tournament_weight": weight,
        "home_elo": team_A_stats['elo'], "away_elo": team_B_stats['elo'],
        "elo_diff": team_A_stats['elo'] - team_B_stats['elo'],
        "home_gf5": team_A_stats['gf5'], "home_ga5": team_A_stats['ga5'],
        "away_gf5": team_B_stats['gf5'], "away_ga5": team_B_stats['ga5'],
        "home_rest_days": team_A_stats['rest'], "away_rest_days": team_B_stats['rest']
    }])[FEATURES]

    row_BA = pd.DataFrame([{
        "neutral": is_neutral, "tournament_weight": weight,
        "home_elo": team_B_stats['elo'], "away_elo": team_A_stats['elo'],
        "elo_diff": team_B_stats['elo'] - team_A_stats['elo'],
        "home_gf5": team_B_stats['gf5'], "home_ga5": team_B_stats['ga5'],
        "away_gf5": team_A_stats['gf5'], "away_ga5": team_A_stats['ga5'],
        "home_rest_days": team_B_stats['rest'], "away_rest_days": team_A_stats['rest']
    }])[FEATURES]

    final_lambda_A = (model_home.predict(row_AB)[0] + model_away.predict(row_BA)[0]) / 2
    final_lambda_B = (model_away.predict(row_AB)[0] + model_home.predict(row_BA)[0]) / 2
    return final_lambda_A, final_lambda_B


def generate_score_matrix(lambda_a, lambda_b, max_goals=10):
    # Increased max_goals from 5 to 10 so probability sum is captured completely
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            matrix[x, y] = poisson.pmf(x, lambda_a) * poisson.pmf(y, lambda_b)
    return matrix


# ==========================================
# 5. Fixture Lookup Helpers
# ==========================================
FIXTURE_NAME_MAP = {
    "IR Iran": "Iran", "Korea Republic": "South Korea", "Türkiye": "Turkey",
    "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao", "USA": "United States",
    "Cape Verde": "Cabo Verde",
}


def map_fixture_name(name):
    return FIXTURE_NAME_MAP.get(name.strip(), name.strip())


def _side_matches(user_input, raw_name):
    u = user_input.strip().lower()
    return u in {raw_name.strip().lower(), map_fixture_name(raw_name).strip().lower()}


def find_fixture(team_a, team_b):
    """Looks up specific 2026 World Cup match info if fixtures.csv exists."""
    if not os.path.exists(FIXTURES_PATH):
        return None
    try:
        fx = pd.read_csv(FIXTURES_PATH)
        for _, row in fx.iterrows():
            if " v " not in str(row.get("teams", "")):
                continue
            left, right = [p.strip() for p in str(row["teams"]).split(" v ")]
            forward = _side_matches(team_a, left) and _side_matches(team_b, right)
            reverse = _side_matches(team_a, right) and _side_matches(team_b, left)
            if forward or reverse:
                return {
                    "group": row.get("group", "Unknown Group"),
                    "stadium": row.get("stadium", "Unknown Stadium"),
                    "date": row.get("date_dt", row.get("date", "Unknown Date"))
                }
    except Exception:
        pass
    return None


# ==========================================
# 6. Interactive Command Line Interface
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("🏆 WORLD CUP 2026 PREDICTION ENGINE READY 🏆")
    print("=" * 50)
    print("Note: Use full country names (e.g., 'United States', 'South Korea')")

    print("\nQuick Reference Teams:")
    print("Algeria, Argentina, Australia, Austria, Belgium, Bosnia and Herzegovina, Brazil,")
    print("Cabo Verde, Canada, Colombia, Congo DR, Croatia, Curaçao, Czechia, Côte d'Ivoire,")
    print("Ecuador, Egypt, England, France, Germany, Ghana, Haiti, IR Iran, Iraq, Japan,")
    print("Jordan, Korea Republic, Mexico, Morocco, Netherlands, New Zealand, Norway,")
    print("Panama, Paraguay, Portugal, Qatar, Saudi Arabia, Scotland, Senegal, South Africa,")
    print("Spain, Sweden, Switzerland, Tunisia, Türkiye, USA, Uruguay, Uzbekistan")

    # Build a case-insensitive lookup dictionary from the historical data
    valid_teams = set(raw_df['home_team']).union(set(raw_df['away_team']))
    valid_teams_lower = {team.lower(): team for team in valid_teams}

    # Add the alias map inspired by the Inspiration Code (lowercased for matching)
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
        """Safely resolves user input to the exact dataset spelling."""
        val = user_input.lower().strip()
        # 1. Check the hardcoded aliases first
        if val in NAME_ALIASES:
            return NAME_ALIASES[val]
        # 2. Check for a case-insensitive match in the actual dataset
        if val in valid_teams_lower:
            return valid_teams_lower[val]
        return None


    while True:
        print("\n" + "-" * 50)
        # We removed .title() so we don't accidentally ruin acronyms
        t1_input = input("Enter Team 1 (or 'quit' to exit): ").strip()
        if t1_input.lower() == 'quit': break

        t2_input = input("Enter Team 2 (or 'quit' to exit): ").strip()
        if t2_input.lower() == 'quit': break

        # Safely resolve the names
        team1 = resolve_team(t1_input)
        team2 = resolve_team(t2_input)

        if not team1:
            print(f"❌ Error: Could not find team '{t1_input}' in historical data. Check spelling.")
            continue
        if not team2:
            print(f"❌ Error: Could not find team '{t2_input}' in historical data. Check spelling.")
            continue

        stats_1 = get_current_stats(team1)
        stats_2 = get_current_stats(team2)

        # Look up fixture details
        m = find_fixture(team1, team2)
        match_info = f"  {m['date']}  ·  {m['group']}  ·  {m['stadium']}" if m else "  Hypothetical / Friendly Match"

        # Predict match!
        l_1, l_2 = predict_symmetric_poisson(stats_1, stats_2, is_neutral=1, weight=4)
        matrix = generate_score_matrix(l_1, l_2)

        t1_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        t2_win = np.sum(np.triu(matrix, 1))

        # Get top 3 exact scorelines
        flat_probs = []
        for i in range(11):  # Adjusted range to 11 to match our max_goals=10
            for j in range(11):
                flat_probs.append((i, j, matrix[i, j]))
        flat_probs.sort(key=lambda x: x[2], reverse=True)

        print("\n📊 MATCH PREDICTION RESULTS 📊")
        print(f"Match: {team1} vs {team2}")
        print(match_info)
        print("-" * 40)
        print(f"Form ({team1}): Elo {stats_1['elo']:.0f} | GF: {stats_1['gf5']:.1f} | GA: {stats_1['ga5']:.1f}")
        print(f"Form ({team2}): Elo {stats_2['elo']:.0f} | GF: {stats_2['gf5']:.1f} | GA: {stats_2['ga5']:.1f}")
        print("-" * 40)
        print(f"🎯 {team1} Expected Goals: {l_1:.2f}")
        print(f"🎯 {team2} Expected Goals: {l_2:.2f}")
        print("-" * 40)
        print(f"🏆 {team1} Win Prob: {t1_win * 100:.1f}%")
        print(f"🤝 Draw Probability: {draw * 100:.1f}%")
        print(f"🏆 {team2} Win Prob: {t2_win * 100:.1f}%")
        print("-" * 40)
        print("🔮 Most Likely Exact Scores:")
        for i in range(3):
            score1, score2, prob = flat_probs[i]
            print(f"   {team1} {score1} - {score2} {team2}  ({prob * 100:.1f}%)")
        print("=" * 50)