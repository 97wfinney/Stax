import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
import joblib
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')


# --- 1. CONFIGURATION ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STAX_MODEL_DIR = MODELS_DIR / "stax_Gemini"

# Backtesting Configuration
BETTING_MINUTES = [10, 20, 30, 45, 60, 75]
STAKE_PER_BET = 10.0

# --- FEATURE LISTS ---
LR_FEATURES = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s', 'home_score', 'away_score', 'score_diff', 'avg_home_odds',
    'avg_away_odds', 'avg_draw_odds', 'std_home_odds', 'std_away_odds',
    'std_draw_odds', 'home_odds_momentum', 'away_odds_momentum',
    'draw_odds_momentum', 'prob_home', 'prob_away', 'prob_draw'
]


# --- 2. DATA PROCESSING FOR BASE MODELS ---

def get_team_names(match_name: str, odds_data: dict) -> Tuple[Optional[str], Optional[str]]:
    if ' vs ' in match_name:
        teams = match_name.split(' vs ')
        return teams[0].strip(), teams[1].strip()
    
    non_draw_keys = [k for odds in odds_data.values() for k in odds if k.lower() != 'draw']
    if len(non_draw_keys) >= 2:
        return non_draw_keys[0], non_draw_keys[1]
    return None, None

def process_match_for_lr(json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
    if not json_data: return None, -1
    
    match_name = json_data[0].get('match', 'Unknown')
    home_team, away_team = get_team_names(match_name, json_data[0].get('odds', {}))
    if not home_team or not away_team: return None, -1

    final_score = json_data[-1]['score'].split(' - ')
    final_home, final_away = int(final_score[0]), int(final_score[1])
    final_outcome = 0 if final_home > final_away else 1 if final_away > final_home else 2

    rows = []
    for entry in json_data:
        h_score, a_score = map(int, entry['score'].split(' - '))
        h_odds, a_odds, d_odds = [], [], []
        for bookie_odds in entry.get('odds', {}).values():
            h_odds.append(bookie_odds.get(home_team))
            a_odds.append(bookie_odds.get(away_team))
            d_odds.append(bookie_odds.get('Draw'))
        
        rows.append({
            'home_score': h_score, 'away_score': a_score,
            'avg_home_odds': np.mean([o for o in h_odds if o]),
            'avg_away_odds': np.mean([o for o in a_odds if o]),
            'avg_draw_odds': np.mean([o for o in d_odds if o]),
        })
        
    df = pd.DataFrame(rows).dropna()
    return df, final_outcome

def process_match_for_xgb(json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
    df, final_outcome = process_match_for_lr(json_data)
    if df is None or df.empty: return None, -1
    
    df['score_diff'] = df['home_score'] - df['away_score']
    df['std_home_odds'] = df['avg_home_odds'].rolling(window=5).std().fillna(0)
    df['std_away_odds'] = df['avg_away_odds'].rolling(window=5).std().fillna(0)
    df['std_draw_odds'] = df['avg_draw_odds'].rolling(window=5).std().fillna(0)
    df['home_odds_momentum'] = df['avg_home_odds'].diff().rolling(window=5).mean()
    df['away_odds_momentum'] = df['avg_away_odds'].diff().rolling(window=5).mean()
    df['draw_odds_momentum'] = df['avg_draw_odds'].diff().rolling(window=5).mean()
    df['prob_home'] = 1 / df['avg_home_odds']
    df['prob_away'] = 1 / df['avg_away_odds']
    df['prob_draw'] = 1 / df['avg_draw_odds']
    df['time_elapsed_s'] = df.index * 40
    
    return df.dropna(), final_outcome

def process_match_for_lstm_sequences(json_data: List[Dict], seq_len=5) -> Tuple[Optional[np.ndarray], int]:
    df, final_outcome = process_match_for_lr(json_data)
    if df is None or len(df) < seq_len: return None, -1

    lstm_feats_df = df[['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].copy()
    lstm_feats_df['score_diff'] = df['home_score'] - df['away_score']
    
    sequences = []
    for i in range(seq_len, len(lstm_feats_df) + 1):
        sequences.append(lstm_feats_df.iloc[i-seq_len:i].values)
        
    return np.array(sequences), final_outcome

# --- NEW HELPER FUNCTION for Backtesting ---
def get_lstm_features_df(json_data: List[Dict]) -> Optional[pd.DataFrame]:
    """Engineers the raw, un-sequenced features for the LSTM model."""
    df, _ = process_match_for_lr(json_data)
    if df is None or df.empty: return None

    lstm_feats_df = df[['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].copy()
    lstm_feats_df['score_diff'] = df['home_score'] - df['away_score']
    return lstm_feats_df


# --- 3. META-MODEL TRAINING ---

def generate_meta_features(data_path: Path, models: Dict, scalers: Dict) -> pd.DataFrame:
    print(f"Generating meta-features from data in: {data_path}")
    json_files = list(data_path.glob('*.json'))
    meta_data = []

    for file in tqdm(json_files, desc="Processing matches for meta-features"):
        with open(file, 'r') as f:
            match_data = json.load(f)

        df_lr, outcome = process_match_for_lr(match_data)
        df_xgb, _ = process_match_for_xgb(match_data)
        sequences_lstm, _ = process_match_for_lstm_sequences(match_data)
        
        if df_lr is None or df_xgb is None or sequences_lstm is None or sequences_lstm.shape[0] == 0:
            continue
            
        common_idx = df_xgb.index.intersection(df_lr.index)
        df_lr = df_lr.loc[common_idx]
        df_xgb = df_xgb.loc[common_idx]
        
        if len(common_idx) <= 5: continue
        lstm_start_idx = common_idx[5:]
        sequences_lstm = sequences_lstm[:len(lstm_start_idx)]
        df_lr = df_lr.loc[lstm_start_idx]
        df_xgb = df_xgb.loc[lstm_start_idx]

        X_lr = scalers['lr'].transform(df_lr[LR_FEATURES])
        preds_lr = models['lr'].predict_proba(X_lr)
        
        X_xgb = scalers['xgb'].transform(df_xgb[XGB_FEATURES])
        preds_xgb = models['xgb'].predict_proba(X_xgb)
        
        flat_lstm = sequences_lstm.reshape(-1, sequences_lstm.shape[2])
        scaled_flat_lstm = scalers['lstm'].transform(flat_lstm)
        X_lstm = scaled_flat_lstm.reshape(sequences_lstm.shape)
        preds_lstm = models['lstm'].predict(X_lstm, verbose=0)

        for i in range(len(preds_lr)):
            row = {
                'p_lr_H': preds_lr[i][0], 'p_lr_A': preds_lr[i][1], 'p_lr_D': preds_lr[i][2],
                'p_xgb_H': preds_xgb[i][0], 'p_xgb_A': preds_xgb[i][1], 'p_xgb_D': preds_xgb[i][2],
                'p_lstm_H': preds_lstm[i][0], 'p_lstm_A': preds_lstm[i][1], 'p_lstm_D': preds_lstm[i][2],
                'final_outcome': outcome
            }
            meta_data.append(row)

    return pd.DataFrame(meta_data)

def train_stax_model():
    print("--- Starting Stax Model Training Pipeline ---")
    
    print("Loading base models and scalers...")
    models = {
        'lr': joblib.load(MODELS_DIR / "logistic_regression_model/logistic_regression_model.joblib"),
        'xgb': joblib.load(MODELS_DIR / "xgboost_model/xgboost_model.joblib"),
        'lstm': tf.keras.models.load_model(MODELS_DIR / "lstm_seq5/lstm_seq5.h5")
    }
    scalers = {
        'lr': joblib.load(MODELS_DIR / "logistic_regression_model/feature_scaler.joblib"),
        'xgb': joblib.load(MODELS_DIR / "xgboost_model/feature_scaler.joblib"),
        'lstm': joblib.load(MODELS_DIR / "lstm_seq5/scaler_seq5.pkl")
    }

    meta_df = generate_meta_features(DATA_DIR / "Training", models, scalers)
    
    if meta_df.empty:
        print("Meta-feature generation failed. No data to train on. Exiting.")
        return None, None, None, None

    print(f"Generated {len(meta_df)} samples for Stax model training.")

    X_meta = meta_df.drop('final_outcome', axis=1)
    y_meta = meta_df['final_outcome']

    stax_scaler = StandardScaler()
    X_meta_scaled = stax_scaler.fit_transform(X_meta)

    stax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    stax_model.fit(X_meta_scaled, y_meta)

    print("\n--- Stax Model Training Evaluation ---")
    preds = stax_model.predict(X_meta_scaled)
    print(f"Accuracy on training set: {accuracy_score(y_meta, preds):.4f}")
    print(classification_report(y_meta, preds, target_names=['Home Win', 'Away Win', 'Draw']))

    STAX_MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(stax_model, STAX_MODEL_DIR / "stax_model.joblib")
    joblib.dump(stax_scaler, STAX_MODEL_DIR / "stax_scaler.joblib")
    print(f"\nStax model and scaler saved to '{STAX_MODEL_DIR}'")

    return stax_model, stax_scaler, models, scalers


# --- 4. BACKTESTING ---

def backtest_stax_model(stax_model, stax_scaler, base_models, base_scalers):
    print("\n--- Starting Stax Model Backtesting ---")
    
    backtest_files = list((DATA_DIR / "Backtest").glob('*.json'))
    all_results = []
    
    for file in tqdm(backtest_files, desc="Backtesting Matches"):
        with open(file, 'r') as f:
            match_data = json.load(f)

        df_xgb_full, outcome = process_match_for_xgb(match_data)
        lstm_features_df = get_lstm_features_df(match_data)
        
        if df_xgb_full is None or lstm_features_df is None or outcome == -1:
            continue
        
        for minute in BETTING_MINUTES:
            time_s = minute * 60
            if time_s > (len(df_xgb_full) - 1) * 40: continue

            target_row = df_xgb_full.iloc[(df_xgb_full['time_elapsed_s'] - time_s).abs().idxmin()]
            
            # --- Generate Base Predictions ---
            # LR
            lr_feats = target_row[LR_FEATURES]
            lr_scaled = base_scalers['lr'].transform(lr_feats.values.reshape(1, -1))
            pred_lr = base_models['lr'].predict_proba(lr_scaled)[0]
            
            # XGB
            xgb_feats = target_row[XGB_FEATURES]
            xgb_scaled = base_scalers['xgb'].transform(xgb_feats.values.reshape(1, -1))
            pred_xgb = base_models['xgb'].predict_proba(xgb_scaled)[0]
            
            # LSTM (Corrected Logic)
            seq_len = 5
            end_idx = target_row.name
            start_idx = end_idx - seq_len + 1
            if start_idx < 0: continue
            
            sequence_df = lstm_features_df.loc[start_idx:end_idx]
            if sequence_df.shape[0] != seq_len: continue
            
            sequence_np = sequence_df.values
            scaled_sequence = base_scalers['lstm'].transform(sequence_np)
            X_lstm = scaled_sequence.reshape(1, seq_len, sequence_np.shape[1])
            pred_lstm = base_models['lstm'].predict(X_lstm, verbose=0)[0]
            
            # --- Generate Stax Prediction ---
            meta_features = np.concatenate([pred_lr, pred_xgb, pred_lstm]).reshape(1, -1)
            meta_scaled = stax_scaler.transform(meta_features)
            stax_probs = stax_model.predict_proba(meta_scaled)[0]
            
            bet_choice = np.argmax(stax_probs)
            odds_map = {0: 'avg_home_odds', 1: 'avg_away_odds', 2: 'avg_draw_odds'}
            bet_odds = target_row[odds_map[bet_choice]]
            
            is_correct = (bet_choice == outcome)
            pnl = (STAKE_PER_BET * bet_odds - STAKE_PER_BET) if is_correct else -STAKE_PER_BET

            all_results.append({'strategy_minute': minute, 'pnl': pnl, 'correct': is_correct})

    if not all_results:
        print("No backtesting results generated.")
        return
        
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby('strategy_minute').agg(
        total_bets=('pnl', 'size'),
        total_pnl=('pnl', 'sum'),
        win_rate=('correct', lambda x: x.mean() * 100)
    )
    summary['total_staked'] = summary['total_bets'] * STAKE_PER_BET
    summary['roi_%'] = (summary['total_pnl'] / summary['total_staked']) * 100
    
    print("\n--- Stax Model Backtest Summary ---")
    print(summary.round(2))


# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    stax_model, stax_scaler, base_models, base_scalers = train_stax_model()
    
    if stax_model:
        backtest_stax_model(stax_model, stax_scaler, base_models, base_scalers)
    
    print("\n--- Stax Project Script Finished ---")