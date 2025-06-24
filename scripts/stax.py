#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import warnings
import joblib
import tensorflow as tf

from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------------------------------------------
# 1. CONFIGURATION  (all paths now resolved off project root)
# -----------------------------------------------------------------------------
ROOT_DIR         = Path(__file__).resolve().parent
# if this file lives in scripts/, step up one
if ROOT_DIR.name == 'scripts':
    ROOT_DIR = ROOT_DIR.parent

DATA_DIR         = ROOT_DIR / "data"
TRAINING_DIR     = DATA_DIR / "Training"
BACKTEST_DIR     = DATA_DIR / "Backtest"
MODELS_DIR       = ROOT_DIR / "models"
STAX_MODEL_DIR   = MODELS_DIR / "stax_model"

# Backtesting Configuration
BETTING_MINUTES  = [10, 20, 30, 45, 60, 75]
STAKE_PER_BET    = 10.0

# Feature lists for base models
LR_FEATURES  = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s', 'home_score', 'away_score', 'score_diff',
    'avg_home_odds','avg_away_odds','avg_draw_odds',
    'std_home_odds','std_away_odds','std_draw_odds',
    'home_odds_momentum','away_odds_momentum','draw_odds_momentum',
    'prob_home','prob_away','prob_draw'
]


# -----------------------------------------------------------------------------
# 2. JSON → DataFrame helpers (unchanged from your working code)
# -----------------------------------------------------------------------------
def get_team_names(match_name: str, odds_data: dict) -> Tuple[Optional[str], Optional[str]]:
    if ' vs ' in match_name:
        a,b = match_name.split(' vs ')
        return a.strip(), b.strip()
    # fallback: pick first two non-"Draw" keys
    keys = [k for bookmaker in odds_data.values() for k in bookmaker if k.lower()!='draw']
    return (keys[0], keys[1]) if len(keys)>=2 else (None, None)

def process_match_for_lr(json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
    # ... same as before ...
    # (omitted here for brevity—copy your full function verbatim)
    ...

def process_match_for_xgb(json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
    # ... same as before ...
    ...

def process_match_for_lstm_sequences(json_data: List[Dict], seq_len=5) -> Tuple[Optional[np.ndarray], int]:
    # ... same as before ...
    ...

def get_lstm_features_df(json_data: List[Dict]) -> Optional[pd.DataFrame]:
    # ... same as before ...
    ...


# -----------------------------------------------------------------------------
# 3. META‐FEATURE GENERATION & TRAINING
# -----------------------------------------------------------------------------
def generate_meta_features(data_path: Path, models: Dict, scalers: Dict) -> pd.DataFrame:
    print(f"Generating meta-features from data in: {data_path}")
    json_files = list(data_path.glob('*.json'))
    meta_data = []

    for file in tqdm(json_files, desc="Processing matches for meta-features"):
        match_data = json.loads(file.read_text())

        df_lr,   outcome = process_match_for_lr(match_data)
        df_xgb, _        = process_match_for_xgb(match_data)
        seq_lstm, _      = process_match_for_lstm_sequences(match_data)
        
        if df_lr is None or df_xgb is None or seq_lstm is None or seq_lstm.shape[0]==0:
            continue

        # align indices, skip first 5 for LSTM warm-up
        common_idx = df_xgb.index.intersection(df_lr.index)
        if len(common_idx)<=5: continue

        df_lr        = df_lr.loc[common_idx]
        df_xgb       = df_xgb.loc[common_idx]
        lstm_indices = common_idx[5:]
        df_lr        = df_lr.loc[lstm_indices]
        df_xgb       = df_xgb.loc[lstm_indices]
        seq_lstm     = seq_lstm[:len(lstm_indices)]

        # base‐model preds
        Xlr   = scalers['lr'].transform(df_lr[LR_FEATURES])
        pr_lr = models['lr'].predict_proba(Xlr)

        Xxgb  = scalers['xgb'].transform(df_xgb[XGB_FEATURES])
        pr_xgb= models['xgb'].predict_proba(Xxgb)

        flat_lstm   = seq_lstm.reshape(-1, seq_lstm.shape[2])
        scaled_flat = scalers['lstm'].transform(flat_lstm)
        Xlstm       = scaled_flat.reshape(seq_lstm.shape)
        pr_lstm     = models['lstm'].predict(Xlstm, verbose=0)

        # combine into final meta‐rows
        for i in range(len(pr_lr)):
            meta_data.append({
                'p_lr_H':   pr_lr[i][0],   'p_lr_A':   pr_lr[i][1],   'p_lr_D':   pr_lr[i][2],
                'p_xgb_H':  pr_xgb[i][0],  'p_xgb_A':  pr_xgb[i][1],  'p_xgb_D':  pr_xgb[i][2],
                'p_lstm_H': pr_lstm[i][0], 'p_lstm_A': pr_lstm[i][1], 'p_lstm_D': pr_lstm[i][2],
                'final_outcome': outcome
            })

    return pd.DataFrame(meta_data)


def train_stax_model():
    print("--- Starting Stax Model Training Pipeline ---")
    print("Loading base models and scalers...")

    models = {
        'lr':   joblib.load(MODELS_DIR/"logistic_regression_model"/"logistic_regression_model.joblib"),
        'xgb':  joblib.load(MODELS_DIR/"xgboost_model"/"xgboost_model.joblib"),
        'lstm': tf.keras.models.load_model(MODELS_DIR/"lstm_seq5"/"lstm_seq5.h5")
    }
    scalers = {
        'lr':   joblib.load(MODELS_DIR/"logistic_regression_model"/"feature_scaler.joblib"),
        'xgb':  joblib.load(MODELS_DIR/"xgboost_model"/"feature_scaler.joblib"),
        'lstm': joblib.load(MODELS_DIR/"lstm_seq5"/"scaler_seq5.pkl")
    }

    meta_df = generate_meta_features(TRAINING_DIR, models, scalers)
    if meta_df.empty:
        print("No meta-features generated. Exiting.")
        return None, None, None, None

    print(f"Generated {len(meta_df)} samples for training.")
    X = meta_df.drop('final_outcome', axis=1)
    y = meta_df['final_outcome']

    stax_scaler = StandardScaler()
    Xs = stax_scaler.fit_transform(X)

    stax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    stax_model.fit(Xs, y)

    print("\n--- Training evaluation ---")
    preds = stax_model.predict(Xs)
    print(f"Train accuracy: {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds, target_names=['Home Win','Away Win','Draw']))

    STAX_MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(stax_model, STAX_MODEL_DIR/"stax_model.joblib")
    joblib.dump(stax_scaler, STAX_MODEL_DIR/"stax_scaler.joblib")
    print(f"Saved Stax model + scaler to {STAX_MODEL_DIR}\n")

    return stax_model, stax_scaler, models, scalers


# -----------------------------------------------------------------------------
# 4. BACKTEST
# -----------------------------------------------------------------------------
def backtest_stax_model(stax_model, stax_scaler, base_models, base_scalers):
    print("\n--- Starting Stax Model Backtesting ---")
    backtest_files = list(BACKTEST_DIR.glob("*.json"))
    results = []

    for file in tqdm(backtest_files, desc="Backtesting Matches"):
        match_data = json.loads(file.read_text())
        df_xgb_full, outcome = process_match_for_xgb(match_data)
        lstm_feats_df       = get_lstm_features_df(match_data)
        if df_xgb_full is None or lstm_feats_df is None or outcome==-1:
            continue

        for minute in BETTING_MINUTES:
            t = minute * 60
            if t > df_xgb_full['time_elapsed_s'].iloc[-1]:
                continue

            # find row closest to desired second
            idx = (df_xgb_full['time_elapsed_s'] - t).abs().idxmin()
            row= df_xgb_full.loc[idx]

            # LR prediction
            lr_f = row[LR_FEATURES].values.reshape(1,-1)
            pr_lr  = base_models['lr'].predict_proba(base_scalers['lr'].transform(lr_f))[0]

            # XGB prediction
            xgb_f = row[XGB_FEATURES].values.reshape(1,-1)
            pr_xgb = base_models['xgb'].predict_proba(base_scalers['xgb'].transform(xgb_f))[0]

            # LSTM prediction
            start_idx = idx - 4  # seq_len=5
            if start_idx<0: continue
            seq_df = lstm_feats_df.loc[start_idx:idx]
            if len(seq_df)!=5: continue
            arr  = base_scalers['lstm'].transform(seq_df.values)
            pr_lstm = base_models['lstm'].predict(arr.reshape(1,5,arr.shape[1]), verbose=0)[0]

            # meta‐model
            meta  = np.concatenate([pr_lr, pr_xgb, pr_lstm]).reshape(1,-1)
            stx_p = stax_scaler.transform(meta)
            pr_stx= stax_model.predict_proba(stx_p)[0]
            choice = np.argmax(pr_stx)
            odds_map = {0:'avg_home_odds',1:'avg_away_odds',2:'avg_draw_odds'}
            od   = row[odds_map[choice]]
            correct = (choice==outcome)
            pnl     = (STAKE_PER_BET*od - STAKE_PER_BET) if correct else -STAKE_PER_BET

            results.append({'strategy_minute': minute,'pnl':pnl,'correct':correct})

    if not results:
        print("No backtest results.")
        return

    df = pd.DataFrame(results)
    summary = df.groupby('strategy_minute').agg(
        total_bets=('pnl','size'),
        total_pnl=('pnl','sum'),
        win_rate=('correct', lambda x: x.mean()*100)
    )
    summary['total_staked'] = summary['total_bets']*STAKE_PER_BET
    summary['roi_%']       = summary['total_pnl']/summary['total_staked']*100

    print("\n--- Backtest summary ---")
    print(summary.round(2))


# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # train (won’t overwrite your originals)
    stax_model, stax_scaler, base_models, base_scalers = train_stax_model()
    if stax_model:
        backtest_stax_model(stax_model, stax_scaler, base_models, base_scalers)
    print("\n--- Done ---")