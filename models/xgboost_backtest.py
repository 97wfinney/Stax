import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import argparse

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. CONFIGURATION ---
MODEL_DIR = Path("./xgboost_model")
TRAINING_DIR = Path("../data/Training")
BACKTEST_DIR = Path("../data/Backtest")
MOMENTUM_WINDOW = 5
STAKE_PER_BET = 10.0
PREDICTION_STRATEGIES = [10, 20, 30, 45, 60]


def get_team_names(match_name: str, odds_data: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Robustly determines home and away team names from match data.
    """
    # 1. Primary Method: Parse "vs" from the match name. Most reliable.
    if ' vs ' in match_name:
        teams = match_name.split(' vs ')
        home_team = teams[0].strip()
        away_team = teams[1].strip()
        if home_team and away_team:
            return home_team, away_team

    # 2. Fallback: Infer from the keys of the odds dictionaries if primary fails.
    all_potential_teams = set()
    for bookmaker_odds in odds_data.values():
        for key in bookmaker_odds.keys():
            # Add only valid, non-draw strings
            if isinstance(key, str) and key.strip() and key.lower() != 'draw':
                all_potential_teams.add(key.strip())
    
    sorted_teams = sorted(list(all_potential_teams))
    if len(sorted_teams) >= 2:
        # This is an assumption, but it's a consistent one.
        return sorted_teams[0], sorted_teams[1]

    # 3. If all else fails, we cannot determine the teams.
    return None, None


def engineer_features(data: List[Dict]) -> Optional[pd.DataFrame]:
    """
    Loads a single match's data, engineers features, and returns a clean DataFrame.
    """
    if not data: return None

    match_name = data[0].get('match', 'Unknown Match')
    
    # Get home/away team names using the robust function
    home_team, away_team = get_team_names(match_name, data[0].get('odds', {}))
    
    # *** CRITICAL VALIDATION ***
    # If we can't reliably determine both teams, skip this entire file.
    if not home_team or not away_team:
        print(f"\nWarning: Could not determine both teams for '{match_name}'. Skipping this file.")
        return None

    final_score_str = data[-1]['score']
    final_home, final_away = map(int, final_score_str.split(' - '))
    if final_home > final_away: final_outcome = 0
    elif final_away > final_home: final_outcome = 1
    else: final_outcome = 2
    
    rows = []
    for i, entry in enumerate(data):
        home_score, away_score = map(int, entry['score'].split(' - '))
        h_odds, a_odds, d_odds = [], [], []
        for bookmaker_odds in entry.get('odds', {}).values():
            h_odds.append(bookmaker_odds.get(home_team))
            a_odds.append(bookmaker_odds.get(away_team))
            d_odds.append(bookmaker_odds.get('Draw'))
        
        h_odds = [o for o in h_odds if o is not None]
        a_odds = [o for o in a_odds if o is not None]
        d_odds = [o for o in d_odds if o is not None]

        rows.append({
            'time_elapsed_s': i * 40, 'home_score': home_score, 'away_score': away_score,
            'score_diff': home_score - away_score,
            'avg_home_odds': np.mean(h_odds) if h_odds else np.nan,
            'avg_away_odds': np.mean(a_odds) if a_odds else np.nan,
            'avg_draw_odds': np.mean(d_odds) if d_odds else np.nan,
            'std_home_odds': np.std(h_odds) if len(h_odds) > 1 else 0,
            'std_away_odds': np.std(a_odds) if len(a_odds) > 1 else 0,
            'std_draw_odds': np.std(d_odds) if len(d_odds) > 1 else 0,
        })

    df = pd.DataFrame(rows)
    df['home_odds_momentum'] = df['avg_home_odds'].diff().rolling(window=MOMENTUM_WINDOW).mean()
    df['away_odds_momentum'] = df['avg_away_odds'].diff().rolling(window=MOMENTUM_WINDOW).mean()
    df['draw_odds_momentum'] = df['avg_draw_odds'].diff().rolling(window=MOMENTUM_WINDOW).mean()
    df['prob_home'] = 1 / df['avg_home_odds']
    df['prob_away'] = 1 / df['avg_away_odds']
    df['prob_draw'] = 1 / df['avg_draw_odds']
    df['final_outcome'] = final_outcome
    return df


def run_training():
    """Processes all training files, trains the model, and saves artifacts."""
    print("--- Running in Training Mode ---")
    json_files = list(TRAINING_DIR.glob('*.json'))
    print(f"Found {len(json_files)} match files in {TRAINING_DIR}.")
    
    all_dfs = [df for file_path in tqdm(json_files, desc="Processing training files") 
               if (df := engineer_features(json.load(open(file_path, 'r')))) is not None]

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.dropna(inplace=True)
    print(f"\nMaster dataset created with {len(master_df)} samples from {len(all_dfs)} valid files.")
    
    features = [col for col in master_df.columns if col != 'final_outcome']
    X = master_df[features]
    y = master_df['final_outcome']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Training XGBoost model on {len(X_train)} samples...")
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, n_estimators=1000,
        learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=50,
        random_state=42
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=100)
    
    print("\n--- Model Evaluation on Validation Set ---")
    preds = model.predict(X_val_scaled)
    print(f"Accuracy: {accuracy_score(y_val, preds):.4f}")

    print(f"\nSaving model and scaler to '{MODEL_DIR}'...")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "xgboost_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")
    print("\nTraining complete! Model artifacts are ready.")


def run_backtesting():
    """Loads a pre-trained model and runs profitability backtest."""
    print("--- Running in Backtest Mode ---")
    backtester = XGBoostProfitabilityBacktester(model_dir=MODEL_DIR, backtest_dir=BACKTEST_DIR)
    backtester.perform_backtest()
    summary = backtester.generate_report()
    if summary is not None:
        backtester.plot_results(summary, save_path="./xgboost_profitability_report.png")


class XGBoostProfitabilityBacktester:
    def __init__(self, model_dir: Path, backtest_dir: Path):
        print(f"Loading model artifacts from: {model_dir}")
        self.model = joblib.load(model_dir / "xgboost_model.joblib")
        self.scaler = joblib.load(model_dir / "feature_scaler.joblib")
        self.backtest_dir = backtest_dir
        self.betting_results = []
        self.model_features = self.model.get_booster().feature_names
        print("Model and scaler loaded successfully.")

    def perform_backtest(self):
        json_files = list(self.backtest_dir.glob('*.json'))
        print(f"\nStarting backtest on {len(json_files)} matches...")
        
        for match_file in tqdm(json_files, desc="Backtesting Progress"):
            with open(match_file, 'r') as f:
                match_data = json.load(f)
            features_df = engineer_features(match_data)

            if features_df is None: continue
            
            actual_outcome = features_df['final_outcome'].iloc[0]

            for pred_minute in PREDICTION_STRATEGIES:
                target_time_s = pred_minute * 60
                betting_row = features_df.iloc[(features_df['time_elapsed_s'] - target_time_s).abs().idxmin()]
                
                if betting_row.isnull().any(): continue
                
                features_to_predict = betting_row[self.model_features]
                features_scaled = self.scaler.transform(features_to_predict.values.reshape(1, -1))
                probabilities = self.model.predict_proba(features_scaled)[0]
                predicted_outcome = np.argmax(probabilities)

                bet_on_map = {0: 'avg_home_odds', 1: 'avg_away_odds', 2: 'avg_draw_odds'}
                bet_odds = betting_row[bet_on_map[predicted_outcome]]
                pnl = (STAKE_PER_BET * bet_odds) - STAKE_PER_BET if predicted_outcome == actual_outcome else -STAKE_PER_BET
                
                self.betting_results.append({
                    'prediction_minute': pred_minute,
                    'bet_odds': bet_odds, 'pnl': pnl,
                    'correct_bet': predicted_outcome == actual_outcome
                })
        print("Backtest processing complete.")

    def generate_report(self):
        if not self.betting_results: return None
        results_df = pd.DataFrame(self.betting_results)
        summary = results_df.groupby('prediction_minute').agg(
            Total_Bets=('pnl', 'size'), Total_PL_Pounds=('pnl', 'sum'),
            Avg_Odds=('bet_odds', 'mean'), Win_Rate_Percent=('correct_bet', lambda x: x.mean() * 100)
        ).reset_index()
        summary['Total_Staked_Pounds'] = summary['Total_Bets'] * STAKE_PER_BET
        summary['ROI_Percent'] = (summary['Total_PL_Pounds'] / summary['Total_Staked_Pounds']) * 100
        
        print("\n" + "="*80)
        print("XGBOOST PROFITABILITY REPORT: COMPARING INDEPENDENT STRATEGIES")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80, "\nEach row represents a completely separate betting strategy.")
        
        column_order = ['prediction_minute', 'Total_Bets', 'Total_Staked_Pounds', 'Total_PL_Pounds',
                        'ROI_Percent', 'Win_Rate_Percent', 'Avg_Odds']
        print(summary[column_order].round(2).to_string(index=False))
        print("="*80)
        return summary

    def plot_results(self, summary_df, save_path=None):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.barplot(x='prediction_minute', y='Total_PL_Pounds', data=summary_df, ax=ax1,
                    palette="viridis", hue='prediction_minute', legend=False)
        ax1.set_xlabel("Betting Strategy (Time in Minutes)", fontsize=12)
        ax1.set_ylabel("Total P&L (£)", fontsize=12)
        ax1.set_title("XGBoost: Performance Comparison", fontsize=16, weight='bold')
        ax2 = ax1.twinx()
        sns.lineplot(x=ax1.get_xticks(), y='ROI_Percent', data=summary_df, ax=ax2,
                     color='r', marker='o', label='ROI (%)')
        ax2.set_ylabel("Return on Investment (ROI %)", color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"\nPlot saved to: {save_path}")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Backtest an XGBoost model.")
    parser.add_argument('--mode', type=str, default='backtest', choices=['train', 'backtest'],
                        help="Set the script to 'train' a new model or 'backtest' an existing one.")
    args = parser.parse_args()

    try:
        if args.mode == 'train':
            run_training()
        else: # default is backtest
            run_backtesting()
    except FileNotFoundError as e:
        print(f"\nERROR: A required file or directory was not found.")
        print(f"Details: {e}")
        print("Please ensure paths are correct and the training script has been run before backtesting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("\n--- FULL TRACEBACK ---")
        traceback.print_exc()
