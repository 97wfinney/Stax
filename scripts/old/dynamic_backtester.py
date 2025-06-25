import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import joblib
import tensorflow as tf
from tqdm import tqdm

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')


# --- 1. CONFIGURATION ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# --- Strategy Configuration ---
# UPDATED: Specify the name of the folder inside /models where your Stax model is saved
STAX_FOLDER_NAME = "stax_Gemini"

# Test thresholds from 50% to 85% in 5% increments
THRESHOLDS_TO_TEST = np.arange(0.50, 0.86, 0.05)
STAKE_PER_BET = 10.0
# Define the window within a match to look for betting opportunities
MIN_BET_MINUTE = 5
MAX_BET_MINUTE = 90

# --- Feature Lists ---
LR_FEATURES = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s', 'home_score', 'away_score', 'score_diff', 'avg_home_odds',
    'avg_away_odds', 'avg_draw_odds', 'std_home_odds', 'std_away_odds',
    'std_draw_odds', 'home_odds_momentum', 'away_odds_momentum',
    'draw_odds_momentum', 'prob_home', 'prob_away', 'prob_draw'
]


# --- 2. DATA PROCESSING HELPERS (Copied from training script for consistency) ---

def get_team_names(match_name, odds_data):
    if ' vs ' in match_name:
        teams = match_name.split(' vs ')
        return teams[0].strip(), teams[1].strip()
    non_draw_keys = [k for odds in odds_data.values() for k in odds if k.lower() != 'draw']
    if len(non_draw_keys) >= 2: return non_draw_keys[0], non_draw_keys[1]
    return None, None

def process_match_for_lr(json_data):
    if not json_data: return None, -1
    match_name = json_data[0].get('match', 'Unknown')
    home_team, away_team = get_team_names(match_name, json_data[0].get('odds', {}))
    if not home_team or not away_team: return None, -1
    final_score = json_data[-1]['score'].split(' - ')
    final_home, final_away = int(final_score[0]), int(final_score[1])
    outcome = 0 if final_home > final_away else 1 if final_away > final_home else 2
    rows = []
    for entry in json_data:
        h_score, a_score = map(int, entry['score'].split(' - '))
        h, a, d = [], [], []
        for odds in entry.get('odds', {}).values():
            h.append(odds.get(home_team)); a.append(odds.get(away_team)); d.append(odds.get('Draw'))
        rows.append({
            'home_score': h_score, 'away_score': a_score,
            'avg_home_odds': np.mean([o for o in h if o]),
            'avg_away_odds': np.mean([o for o in a if o]),
            'avg_draw_odds': np.mean([o for o in d if o]),
        })
    return pd.DataFrame(rows).dropna(), outcome

def process_match_for_xgb(json_data):
    df, outcome = process_match_for_lr(json_data)
    if df is None or df.empty: return None, -1
    df['score_diff'] = df['home_score'] - df['away_score']
    df['std_home_odds'] = df['avg_home_odds'].rolling(5).std().fillna(0)
    df['std_away_odds'] = df['avg_away_odds'].rolling(5).std().fillna(0)
    df['std_draw_odds'] = df['avg_draw_odds'].rolling(5).std().fillna(0)
    df['home_odds_momentum'] = df['avg_home_odds'].diff().rolling(5).mean()
    df['away_odds_momentum'] = df['avg_away_odds'].diff().rolling(5).mean()
    df['draw_odds_momentum'] = df['avg_draw_odds'].diff().rolling(5).mean()
    df['prob_home'] = 1/df['avg_home_odds']; df['prob_away'] = 1/df['avg_away_odds']; df['prob_draw'] = 1/df['avg_draw_odds']
    df['time_elapsed_s'] = df.index * 40
    return df.dropna(), outcome

def get_lstm_features_df(json_data):
    df, _ = process_match_for_lr(json_data)
    if df is None or df.empty: return None
    lstm_feats = df[['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].copy()
    lstm_feats['score_diff'] = df['home_score'] - df['away_score']
    return lstm_feats


# --- 3. DYNAMIC BACKTESTING CORE FUNCTION ---

def run_dynamic_backtest():
    """
    Loads all models and runs a backtest simulating a dynamic betting strategy
    based on a range of confidence thresholds.
    """
    print("--- Starting Dynamic Threshold Backtest ---")
    print(f"Testing thresholds: {[f'{t:.0%}' for t in THRESHOLDS_TO_TEST]}")

    # 1. Load all models and scalers
    print("Loading all models and scalers...")
    try:
        # UPDATED: Paths now point to the specific subfolder
        stax_model_file = MODELS_DIR / STAX_FOLDER_NAME / "stax_model.joblib"
        stax_scaler_file = MODELS_DIR / STAX_FOLDER_NAME / "stax_scaler.joblib"
        
        print(f"Attempting to load Stax model: {stax_model_file}")
        print(f"Attempting to load Stax scaler: {stax_scaler_file}")

        models = {
            'lr': joblib.load(MODELS_DIR / "logistic_regression_model/logistic_regression_model.joblib"),
            'xgb': joblib.load(MODELS_DIR / "xgboost_model/xgboost_model.joblib"),
            'lstm': tf.keras.models.load_model(MODELS_DIR / "lstm_seq5/lstm_seq5.h5"),
            'stax': joblib.load(stax_model_file)
        }
        scalers = {
            'lr': joblib.load(MODELS_DIR / "logistic_regression_model/feature_scaler.joblib"),
            'xgb': joblib.load(MODELS_DIR / "xgboost_model/feature_scaler.joblib"),
            'lstm': joblib.load(MODELS_DIR / "lstm_seq5/scaler_seq5.pkl"),
            'stax': joblib.load(stax_scaler_file)
        }
    except FileNotFoundError as e:
        print(f"\nError loading model file: {e}")
        print("Please ensure the model and scaler files exist and the names/paths are correct.")
        return

    backtest_files = list((DATA_DIR / "Backtest").glob('*.json'))
    all_results = []

    # 2. Loop through each match in the backtest set
    for file in tqdm(backtest_files, desc="Processing Matches"):
        with open(file, 'r') as f:
            match_data = json.load(f)

        df_xgb_full, outcome = process_match_for_xgb(match_data)
        lstm_features_df = get_lstm_features_df(match_data)

        if df_xgb_full is None or lstm_features_df is None or outcome == -1:
            continue

        # 3. Test each threshold strategy for the current match
        for threshold in THRESHOLDS_TO_TEST:
            bet_placed_for_this_threshold = False
            
            # 4. Iterate through the time of the match
            min_index = (MIN_BET_MINUTE * 60) // 40
            max_index = (MAX_BET_MINUTE * 60) // 40
            
            for idx, live_row in df_xgb_full.iloc[min_index:max_index].iterrows():
                # --- Generate live predictions ---
                seq_len = 5
                start_idx = idx - seq_len + 1
                if start_idx < 0: continue
                
                lr_scaled = scalers['lr'].transform(live_row[LR_FEATURES].values.reshape(1, -1))
                pred_lr = models['lr'].predict_proba(lr_scaled)[0]

                xgb_scaled = scalers['xgb'].transform(live_row[XGB_FEATURES].values.reshape(1, -1))
                pred_xgb = models['xgb'].predict_proba(xgb_scaled)[0]

                sequence_df = lstm_features_df.loc[start_idx:idx]
                if sequence_df.shape[0] != seq_len: continue
                sequence_np = sequence_df.values
                scaled_seq = scalers['lstm'].transform(sequence_np)
                X_lstm = scaled_seq.reshape(1, seq_len, sequence_np.shape[1])
                pred_lstm = models['lstm'].predict(X_lstm, verbose=0)[0]

                meta_feats = np.concatenate([pred_lr, pred_xgb, pred_lstm]).reshape(1, -1)
                meta_scaled = scalers['stax'].transform(meta_feats)
                stax_probs = models['stax'].predict_proba(meta_scaled)[0]
                max_prob = np.max(stax_probs)

                # --- The Dynamic Bet Logic ---
                if max_prob >= threshold:
                    bet_choice = np.argmax(stax_probs)
                    odds_map = {0: 'avg_home_odds', 1: 'avg_away_odds', 2: 'avg_draw_odds'}
                    bet_odds = live_row[odds_map[bet_choice]]
                    
                    is_correct = (bet_choice == outcome)
                    pnl = (STAKE_PER_BET * bet_odds - STAKE_PER_BET) if is_correct else -STAKE_PER_BET

                    all_results.append({
                        'threshold': threshold,
                        'pnl': pnl,
                        'correct': is_correct,
                        'bet_minute': int(live_row['time_elapsed_s'] / 60)
                    })
                    bet_placed_for_this_threshold = True
                    break
            
    # 5. Summarize and display final results
    if not all_results:
        print("\nNo bets were placed in the backtest. Try lowering thresholds or checking data.")
        return

    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby('threshold').agg(
        total_bets=('pnl', 'size'),
        total_pnl=('pnl', 'sum'),
        win_rate=('correct', lambda x: x.mean() * 100),
        avg_bet_minute=('bet_minute', 'mean')
    )
    summary['total_staked'] = summary['total_bets'] * STAKE_PER_BET
    summary['roi_%'] = (summary['total_pnl'] / summary['total_staked']) * 100
    
    print("\n--- Dynamic Threshold Backtest Summary ---")
    print(summary.round(2))
    print("\n--- Analysis ---")
    if summary['total_pnl'].max() > 0:
        best_strategy = summary.loc[summary['total_pnl'].idxmax()]
        print(f"The most profitable strategy was using a threshold of {best_strategy.name:.0%}.")
        print(f"It placed {int(best_strategy['total_bets'])} bets with an ROI of {best_strategy['roi_%']:.2f}%.")
    else:
        print("No profitable strategy was found in this backtest.")


if __name__ == "__main__":
    run_dynamic_backtest()
    print("\n--- Dynamic Backtester Script Finished ---")