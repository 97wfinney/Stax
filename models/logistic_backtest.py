import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. CONFIGURATION ---
MODEL_DIR = Path("./logistic_regression_model")
BACKTEST_DIR = Path("../data/Backtest")
STAKE_PER_BET = 10.0
PREDICTION_STRATEGIES = [10, 20, 30, 45, 60]

# --- 2. FEATURE ENGINEERING (must be identical to training) ---
def get_team_names(match_name: str, odds_data: dict) -> Tuple[Optional[str], Optional[str]]:
    if ' vs ' in match_name:
        teams = match_name.split(' vs ')
        home_team, away_team = teams[0].strip(), teams[1].strip()
        if home_team and away_team: return home_team, away_team
    all_potential_teams = {k.strip() for d in odds_data.values() for k in d if isinstance(k, str) and k.strip().lower() != 'draw'}
    sorted_teams = sorted(list(all_potential_teams))
    if len(sorted_teams) >= 2: return sorted_teams[0], sorted_teams[1]
    return None, None

def engineer_features_for_lr(data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
    if not data: return None, -1
    match_name = data[0].get('match', 'Unknown Match')
    home_team, away_team = get_team_names(match_name, data[0].get('odds', {}))
    if not home_team or not away_team: return None, -1

    final_home, final_away = map(int, data[-1]['score'].split(' - '))
    if final_home > final_away: actual_outcome = 0
    elif final_away > final_home: actual_outcome = 1
    else: actual_outcome = 2
    
    rows = []
    for entry in data:
        home_score, away_score = map(int, entry['score'].split(' - '))
        h_odds, a_odds, d_odds = [], [], []
        for bookmaker_odds in entry.get('odds', {}).values():
            h_odds.append(bookmaker_odds.get(home_team))
            a_odds.append(bookmaker_odds.get(away_team))
            d_odds.append(bookmaker_odds.get('Draw'))
        
        h_odds, a_odds, d_odds = [o for o in h_odds if o], [o for o in a_odds if o], [o for o in d_odds if o]
        rows.append({
            'time_elapsed_s': len(rows) * 40,
            'home_score': home_score, 'away_score': away_score,
            'avg_home_odds': np.mean(h_odds) if h_odds else np.nan,
            'avg_away_odds': np.mean(a_odds) if a_odds else np.nan,
            'avg_draw_odds': np.mean(d_odds) if d_odds else np.nan,
        })
    df = pd.DataFrame(rows).dropna()
    return df, actual_outcome

# --- 3. BACKTESTING CLASS ---
class ProfitabilityBacktester:
    def __init__(self, model_dir: Path, backtest_dir: Path):
        print(f"Loading model artifacts from: {model_dir}")
        self.model: LogisticRegression = joblib.load(model_dir / "logistic_regression_model.joblib")
        self.scaler: StandardScaler = joblib.load(model_dir / "feature_scaler.joblib")
        self.backtest_dir = backtest_dir
        self.results = []
        self.features = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
        print("Model and scaler loaded successfully.")

    def run(self):
        json_files = list(self.backtest_dir.glob('*.json'))
        print(f"\nStarting backtest on {len(json_files)} matches...")
        
        for match_file in tqdm(json_files, desc="Backtesting Progress"):
            with open(match_file, 'r') as f:
                match_data = json.load(f)
            
            features_df, actual_outcome = engineer_features_for_lr(match_data)
            if features_df is None or features_df.empty: continue

            for pred_minute in PREDICTION_STRATEGIES:
                target_time_s = pred_minute * 60
                if features_df.empty or target_time_s > features_df['time_elapsed_s'].max(): continue
                
                row = features_df.iloc[(features_df['time_elapsed_s'] - target_time_s).abs().idxmin()]
                
                features_to_predict = row[self.features]
                features_scaled = self.scaler.transform(features_to_predict.values.reshape(1, -1))
                probabilities = self.model.predict_proba(features_scaled)[0]
                predicted_outcome = np.argmax(probabilities)

                odds_map = {0: 'avg_home_odds', 1: 'avg_away_odds', 2: 'avg_draw_odds'}
                bet_odds = row[odds_map[predicted_outcome]]
                pnl = (STAKE_PER_BET * bet_odds) - STAKE_PER_BET if predicted_outcome == actual_outcome else -STAKE_PER_BET
                
                self.results.append({
                    'strategy': pred_minute, 'pnl': pnl, 'correct': predicted_outcome == actual_outcome,
                    'bet_odds': bet_odds
                })

    def generate_report_and_plot(self):
        if not self.results:
            print("No results to report.")
            return

        df_res = pd.DataFrame(self.results)
        summary = df_res.groupby('strategy').agg(
            total_bets=('pnl', 'size'),
            total_pl=('pnl', 'sum'),
            win_rate=('correct', lambda x: x.mean() * 100),
            avg_odds=('bet_odds', 'mean')
        ).reset_index()
        summary['roi'] = (summary['total_pl'] / (summary['total_bets'] * STAKE_PER_BET)) * 100

        print("\n" + "="*80)
        print("LOGISTIC REGRESSION PROFITABILITY REPORT")
        print("="*80)
        print(summary.round(2).to_string(index=False))
        print("="*80)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='strategy', y='total_pl', data=summary, ax=ax1, palette="viridis", hue='strategy', legend=False)
        ax1.set_xlabel("Betting Strategy (Time in Minutes)")
        ax1.set_ylabel("Total P&L (£)")
        ax2 = ax1.twinx()
        sns.lineplot(x=ax1.get_xticks(), y='roi', data=summary, ax=ax2, color='r', marker='o', label='ROI (%)')
        ax2.set_ylabel("ROI (%)", color='r')
        ax2.legend(loc='upper right')
        plt.title("Logistic Regression: Performance Comparison")
        plt.show()

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        backtester = ProfitabilityBacktester(model_dir=MODEL_DIR, backtest_dir=BACKTEST_DIR)
        backtester.run()
        backtester.generate_report_and_plot()
    except FileNotFoundError:
        print(f"ERROR: Model artifacts not found in '{MODEL_DIR}'.")
        print("Please run the 'train_logistic_regression.py' script first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
