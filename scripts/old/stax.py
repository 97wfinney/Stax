#!/usr/bin/env python3
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------------------------------------------
# 0. SILENCE TF WARNINGS
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# -----------------------------------------------------------------------------
# 1. PATH CONFIGURATION
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# if this file sits in 'scripts/', step up one to project root
ROOT_DIR   = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'scripts' else SCRIPT_DIR

TRAINING_DIR   = ROOT_DIR / "data" / "Training"
BACKTEST_DIR   = ROOT_DIR / "data" / "Backtest"
MODELS_DIR     = ROOT_DIR / "models"
STAX_MODEL_DIR = MODELS_DIR / "stax_GPT"

# ensure stax_model folder exists
STAX_MODEL_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS
# -----------------------------------------------------------------------------
# Minutes at which to backtest
BETTING_MINUTES = [10, 20, 30, 45, 60, 75]
STAKE_PER_BET   = 10.0

# Features for base models
LR_FEATURES  = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s','home_score','away_score','score_diff',
    'avg_home_odds','avg_away_odds','avg_draw_odds',
    'std_home_odds','std_away_odds','std_draw_odds',
    'home_odds_momentum','away_odds_momentum','draw_odds_momentum',
    'prob_home','prob_away','prob_draw'
]

# -----------------------------------------------------------------------------
# 3. HELPERS: JSON → Tabular for Base Models
# -----------------------------------------------------------------------------
def get_team_names(match_name: str, odds_data: dict) -> Tuple[Optional[str],Optional[str]]:
    if ' vs ' in match_name:
        a,b = match_name.split(' vs ')
        return a.strip(), b.strip()
    # fallback: pick first two non-Draw keys
    keys = [k for book in odds_data.values() for k in book if k.lower()!='draw']
    return (keys[0], keys[1]) if len(keys)>=2 else (None,None)

def process_match_for_lr(json_data: List[dict]) -> Tuple[Optional[pd.DataFrame],int]:
    if not json_data:
        return None, -1

    match_name = json_data[0].get('match','Unknown')
    home_team, away_team = get_team_names(match_name, json_data[0].get('odds',{}))
    if not home_team or not away_team:
        return None, -1

    rows = []
    for entry in json_data:
        # parse scores
        try:
            h_s,a_s = map(int, entry['score'].split(' - '))
        except:
            continue

        # collect each bookie’s odds
        bookies = entry.get('odds',{})
        h_odds = [b.get(home_team)   for b in bookies.values() if b.get(home_team)   is not None]
        a_odds = [b.get(away_team)   for b in bookies.values() if b.get(away_team)   is not None]
        d_odds = [b.get('Draw')      for b in bookies.values() if b.get('Draw')        is not None]
        if not (h_odds and a_odds and d_odds):
            continue

        rows.append({
            'home_score':     h_s,
            'away_score':     a_s,
            'avg_home_odds':  np.mean(h_odds),
            'avg_away_odds':  np.mean(a_odds),
            'avg_draw_odds':  np.mean(d_odds),
        })

    if not rows:
        return None, -1

    df = pd.DataFrame(rows)
    fh, fa = df['home_score'].iloc[-1], df['away_score'].iloc[-1]
    outcome = 0 if fh>fa else (1 if fa>fh else 2)
    return df, outcome

def process_match_for_xgb(json_data: List[dict]) -> Tuple[Optional[pd.DataFrame],int]:
    df, outcome = process_match_for_lr(json_data)
    if df is None or df.empty:
        return None, -1

    df = df.copy()
    df['score_diff'] = df['home_score'] - df['away_score']
    # rolling & momentum
    df['std_home_odds']    = df['avg_home_odds'].rolling(5).std().fillna(0)
    df['std_away_odds']    = df['avg_away_odds'].rolling(5).std().fillna(0)
    df['std_draw_odds']    = df['avg_draw_odds'].rolling(5).std().fillna(0)
    df['home_odds_momentum']= df['avg_home_odds'].diff().rolling(5).mean().fillna(0)
    df['away_odds_momentum']= df['avg_away_odds'].diff().rolling(5).mean().fillna(0)
    df['draw_odds_momentum']= df['avg_draw_odds'].diff().rolling(5).mean().fillna(0)
    # implied probabilities
    df['prob_home']  = 1/df['avg_home_odds']
    df['prob_away']  = 1/df['avg_away_odds']
    df['prob_draw']  = 1/df['avg_draw_odds']
    # time stamp assuming 40s intervals
    df['time_elapsed_s']= np.arange(len(df))*40

    return df.dropna(), outcome

def process_match_for_lstm_sequences(json_data: List[dict], seq_len=5) -> Tuple[Optional[np.ndarray],int]:
    df, outcome = process_match_for_lr(json_data)
    if df is None or len(df) < seq_len:
        return None, -1

    feats = df[['avg_home_odds','avg_away_odds','avg_draw_odds']].copy()
    feats['score_diff'] = df['home_score'] - df['away_score']

    seqs = []
    for i in range(seq_len, len(feats)+1):
        seqs.append(feats.iloc[i-seq_len:i].values)
    return np.array(seqs), outcome

def get_lstm_features_df(json_data: List[dict]) -> Optional[pd.DataFrame]:
    df, _ = process_match_for_lr(json_data)
    if df is None or df.empty:
        return None
    out = df[['avg_home_odds','avg_away_odds','avg_draw_odds']].copy()
    out['score_diff'] = df['home_score'] - df['away_score']
    return out

# -----------------------------------------------------------------------------
# 4. META-FEATURE GENERATION
# -----------------------------------------------------------------------------
def generate_meta_features(data_dir: Path,
                           models: Dict[str,any],
                           scalers: Dict[str,any]) -> pd.DataFrame:
    print(f"Generating meta-features from {data_dir}")
    files = list(data_dir.glob("*.json"))
    rows  = []

    for fp in tqdm(files, desc="Matches"):
        raw = json.loads(fp.read_text())
        df_lr,   outcome = process_match_for_lr(raw)
        df_xgb,  _       = process_match_for_xgb(raw)
        seqs,    _       = process_match_for_lstm_sequences(raw)
        if df_lr is None or df_xgb is None or seqs is None or seqs.shape[0]==0:
            continue

        # align to indices where all three exist, skip first seq_len rows
        idx = df_xgb.index.intersection(df_lr.index)
        if len(idx) <= seqs.shape[1]:
            continue

        df_lr = df_lr.loc[idx]
        df_xgb= df_xgb.loc[idx]
        start = seqs.shape[1]  # 5
        df_lr = df_lr.iloc[start:]
        df_xgb= df_xgb.iloc[start:]
        seqs   = seqs[:len(df_lr)]

        # base predictions
        pred_lr   = models['lr'].predict_proba(scalers['lr'].transform(df_lr[LR_FEATURES]))
        pred_xgb  = models['xgb'].predict_proba(scalers['xgb'].transform(df_xgb[XGB_FEATURES]))
        flat      = seqs.reshape(-1, seqs.shape[2])
        flat_scl  = scalers['lstm'].transform(flat)
        pred_lstm = models['lstm'].predict(flat_scl.reshape(seqs.shape), verbose=0)

        for i in range(len(pred_lr)):
            rows.append({
                'p_lr_H':   pred_lr[i][0], 'p_lr_A':   pred_lr[i][1], 'p_lr_D':   pred_lr[i][2],
                'p_xgb_H':  pred_xgb[i][0],'p_xgb_A':  pred_xgb[i][1],'p_xgb_D':  pred_xgb[i][2],
                'p_lstm_H': pred_lstm[i][0],'p_lstm_A': pred_lstm[i][1],'p_lstm_D': pred_lstm[i][2],
                'final_outcome': outcome
            })

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 5. TRAIN Stax META-MODEL
# -----------------------------------------------------------------------------
def train_stax_model():
    print("\n=== Training Stax Meta-Model ===")
    # load base models + scalers
    models  = {
        'lr':   joblib.load(MODELS_DIR/"logistic_regression_model"/"logistic_regression_model.joblib"),
        'xgb':  joblib.load(MODELS_DIR/"xgboost_model"/"xgboost_model.joblib"),
        'lstm': tf.keras.models.load_model(MODELS_DIR/"lstm_seq5"/"lstm_seq5.h5")
    }
    scalers = {
        'lr':   joblib.load(MODELS_DIR/"logistic_regression_model"/"feature_scaler.joblib"),
        'xgb':  joblib.load(MODELS_DIR/"xgboost_model"/"feature_scaler.joblib"),
        'lstm': joblib.load(MODELS_DIR/"lstm_seq5"/"scaler_seq5.pkl")
    }

    meta = generate_meta_features(TRAINING_DIR, models, scalers)
    if meta.empty:
        print("✖ No meta-features generated; aborting.")
        return None, None, None, None

    print(f"✔ Generated {len(meta)} samples")
    X = meta.drop('final_outcome', axis=1)
    y = meta['final_outcome']

    stax_scaler = StandardScaler().fit(X)
    Xs          = stax_scaler.transform(X)

    stax_model  = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000,random_state=42)
    stax_model.fit(Xs, y)

    print("\n-- Training metrics --")
    preds = stax_model.predict(Xs)
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds, target_names=['Home Win','Away Win','Draw']))

    # save
    joblib.dump(stax_model,  STAX_MODEL_DIR/"stax_model.joblib")
    joblib.dump(stax_scaler, STAX_MODEL_DIR/"stax_scaler.joblib")
    print(f"✔ Saved Stax model & scaler to {STAX_MODEL_DIR}")

    return stax_model, stax_scaler, models, scalers

# -----------------------------------------------------------------------------
# 6. BACKTEST
# -----------------------------------------------------------------------------
def backtest_stax_model(stax_model, stax_scaler, base_models, base_scalers):
    print("\n=== Backtesting Stax Model ===")
    files   = list(BACKTEST_DIR.glob("*.json"))
    results = []

    for fp in tqdm(files, desc="Backtest"):
        raw = json.loads(fp.read_text())
        df_xgb, outcome = process_match_for_xgb(raw)
        lstm_df        = get_lstm_features_df(raw)
        if df_xgb is None or lstm_df is None or outcome<0:
            continue

        for minute in BETTING_MINUTES:
            tsec = minute * 60
            if tsec > df_xgb['time_elapsed_s'].iloc[-1]:
                continue

            # pick the row closest in time
            idx = (df_xgb['time_elapsed_s'] - tsec).abs().idxmin()
            row = df_xgb.loc[idx]

            # LR
            lr_feat = row[LR_FEATURES].values.reshape(1,-1)
            pr_lr   = base_models['lr'].predict_proba(base_scalers['lr'].transform(lr_feat))[0]

            # XGB
            xgb_feat= row[XGB_FEATURES].values.reshape(1,-1)
            pr_xgb  = base_models['xgb'].predict_proba(base_scalers['xgb'].transform(xgb_feat))[0]

            # LSTM
            seq_len = lstm_df.shape[1] if hasattr(lstm_df,'shape') else 5
            start   = idx - (seq_len-1)
            if start<0: continue
            seq     = lstm_df.loc[start:idx].values
            pr_lstm = base_models['lstm'].predict(
                base_scalers['lstm'].transform(seq).reshape(1,seq_len,seq.shape[1]), verbose=0
            )[0]

            # meta
            meta_feat = np.concatenate([pr_lr, pr_xgb, pr_lstm]).reshape(1,-1)
            pr_stx    = stax_model.predict_proba(stax_scaler.transform(meta_feat))[0]
            choice    = int(np.argmax(pr_stx))
            odds_map  = {0:'avg_home_odds',1:'avg_away_odds',2:'avg_draw_odds'}
            od        = row[odds_map[choice]]
            correct   = (choice == outcome)
            pnl       = (STAKE_PER_BET*od - STAKE_PER_BET) if correct else -STAKE_PER_BET

            results.append({
                'minute': minute,
                'pnl':     pnl,
                'correct': correct
            })

    if not results:
        print("✖ No backtest bets placed.")
        return

    df = pd.DataFrame(results)
    sum_df = df.groupby('minute').agg(
        total_bets=('pnl','size'),
        total_pnl=('pnl','sum'),
        win_rate=('correct', lambda x: x.mean()*100)
    )
    sum_df['total_staked'] = sum_df['total_bets'] * STAKE_PER_BET
    sum_df['roi_%']        = sum_df['total_pnl'] / sum_df['total_staked'] * 100

    print("\n-- Backtest Summary --")
    print(sum_df.round(2))

# -----------------------------------------------------------------------------
# 7. MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    stax_model, stax_scaler, bm, bs = train_stax_model()
    if stax_model:
        backtest_stax_model(stax_model, stax_scaler, bm, bs)
    print("\nDone.\n")