#!/usr/bin/env python3


import json
import argparse
# Backtest options: use --kelly to specify a fractional Kelly stake (0 = flat stake)
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- PATHS & CONSTANTS ---
ROOT_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = ROOT_DIR.parent / "data" / "Backtest"
STAX_MODEL_DIR = ROOT_DIR.parent / "models" / "stax3"

LR_FEATURES = ['home_score','away_score','avg_home_odds','avg_away_odds','avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s','home_score','away_score','score_diff',
    'avg_home_odds','avg_away_odds','avg_draw_odds',
    'std_home_odds','std_away_odds','std_draw_odds',
    'home_odds_momentum','away_odds_momentum','draw_odds_momentum',
    'prob_home','prob_away','prob_draw'
]
LSTM_FEATURES = ['avg_home_odds','avg_away_odds','avg_draw_odds','score_diff']
SEQUENCE_LENGTH = 5
HALF_TIME_BREAK = 15*60

# --- FEATURE EXTRACTION ---
def get_match_features(data):
    if not data:
        return None, -1
    home, away = data[0]['match'].split(' vs ')
    fh, fa = map(int, data[-1]['score'].split(' - '))
    outcome = 0 if fh>fa else (1 if fa>fh else 2)
    rows=[]
    for i,e in enumerate(data):
        hs, aw = map(int, e['score'].split(' - '))
        odds = e['odds']
        ho = [o.get(home) for o in odds.values() if o.get(home)]
        ao = [o.get(away) for o in odds.values() if o.get(away)]
        do = [o.get('Draw') for o in odds.values() if o.get('Draw')]
        rows.append({
            'time_elapsed_s': i*40,
            'home_score': hs, 'away_score': aw,
            'avg_home_odds': np.mean(ho) if ho else np.nan,
            'avg_away_odds': np.mean(ao) if ao else np.nan,
            'avg_draw_odds': np.mean(do) if do else np.nan
        })
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return None, outcome
    df['score_diff'] = df['home_score'] - df['away_score']
    # rolling stats
    for col in ['avg_home_odds','avg_away_odds','avg_draw_odds']:
        key = col.split('_')[1]
        df[f'std_{key}_odds'] = df[col].rolling(5).std().fillna(0)
        df[f'{key}_odds_momentum'] = df[col].diff().rolling(5).mean().fillna(0)
    df['prob_home'] = 1/df['avg_home_odds']
    df['prob_away'] = 1/df['avg_away_odds']
    df['prob_draw'] = 1/df['avg_draw_odds']
    return df, outcome

# --- ANALYZER ---
class SingleBetTester:
    def __init__(self, threshold, flat_stake, initial_bank, kelly_fraction):
        self.threshold = threshold
        self.flat_stake = flat_stake
        self.initial_bank = initial_bank
        self.kelly_fraction = kelly_fraction
        self._load_models()

    def _load_models(self):
        # Try loading config from stax3, otherwise fallback to stax_kfold
        config_path = STAX_MODEL_DIR / 'stax_config.joblib'
        if not config_path.exists():
            config_path = STAX_MODEL_DIR.parent / 'stax_kfold' / 'stax_config.joblib'
        cfg = joblib.load(config_path)
        self.weights = cfg.get('model_weights', [0.34, 0.33, 0.33])
        # load logistic, xgboost, lstm
        self.models = {}
        self.scalers = {}
        lr_dir = STAX_MODEL_DIR / 'logistic_regression'
        self.models['lr'] = joblib.load(lr_dir / 'model.joblib')
        self.scalers['lr'] = joblib.load(lr_dir / 'scaler.joblib')
        xgb_dir = STAX_MODEL_DIR / 'xgboost'
        self.models['xgb'] = joblib.load(xgb_dir / 'model.joblib')
        self.scalers['xgb'] = joblib.load(xgb_dir / 'scaler.joblib')
        lstm_dir = STAX_MODEL_DIR / 'lstm'
        self.models['lstm'] = tf.keras.models.load_model(str(lstm_dir / 'model.h5'))
        self.scalers['lstm'] = joblib.load(lstm_dir / 'scaler.pkl')

    def _predict_probs(self, df):
        lr_p = self.models['lr'].predict_proba(self.scalers['lr'].transform(df[LR_FEATURES]))
        xgb_p = self.models['xgb'].predict_proba(self.scalers['xgb'].transform(df[XGB_FEATURES]))
        # LSTM
        seqs = []
        for i in range(len(df)):
            seq = df[LSTM_FEATURES].iloc[max(0,i-SEQUENCE_LENGTH+1):i+1].values
            if len(seq) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH-len(seq), seq.shape[1]))
                seq = np.vstack([pad, seq])
            seqs.append(seq)
        arr = np.array(seqs)
        flat = arr.reshape(-1, arr.shape[-1])
        scaled = self.scalers['lstm'].transform(flat).reshape(arr.shape)
        lstm_p = self.models['lstm'].predict(scaled, verbose=0)
        # ensemble
        ens = self.weights[0]*lr_p + self.weights[1]*xgb_p + self.weights[2]*lstm_p
        return ens


    def run(self, verbose=True):
        bank = self.initial_bank
        history = []
        for f in sorted(BACKTEST_DIR.glob('*.json')):
            data = json.load(open(f))
            df, outcome = get_match_features(data)
            if df is None:
                continue
            probs = self._predict_probs(df)
            for i, row in df.iterrows():
                if i < SEQUENCE_LENGTH:
                    continue
                p = probs[i]
                conf = float(p.max()); idx = int(p.argmax())
                if conf < self.threshold:
                    continue
                # determine the selected odds before any staking logic
                odds = float(row[['avg_home_odds','avg_away_odds','avg_draw_odds']].iloc[idx])
                if self.kelly_fraction > 0:
                    # fractional Kelly criterion
                    # skip cases where odds == 1 to avoid zero division
                    if odds <= 1.0:
                        continue
                    edge = conf * odds - 1
                    frac = edge / (odds - 1)
                    stake = frac * self.kelly_fraction * bank
                    # clamp to available bank
                    stake = max(0.0, min(stake, bank))
                    if stake < 0.01:
                        continue
                else:
                    stake = self.flat_stake
                pnl = stake*(odds-1) if idx==outcome else -stake
                bank += pnl
                minute = int(((row.time_elapsed_s - HALF_TIME_BREAK) if row.time_elapsed_s>45*60 else row.time_elapsed_s)//60)
                history.append({
                    'file': f.name,
                    'minute': minute,
                    'state': f"{int(row.home_score)}-{int(row.away_score)}",
                    'pred': ['Home','Away','Draw'][idx],
                    'actual': ['Home','Away','Draw'][outcome],
                    'conf': conf,
                    'odds': odds,
                    'stake': stake,
                    'pnl': pnl,
                    'bank': bank
                })
                break  # one bet per match
        if not history:
            if verbose:
                print("No bets placed.")
            return pd.DataFrame(), self.initial_bank, 0.0, 0.0, 0.0, 0.0
        dfh = pd.DataFrame(history)
        # summary
        total_pnl = dfh['pnl'].sum()
        total_staked = dfh['stake'].sum()
        roi = (total_pnl/total_staked*100) if total_staked>0 else 0
        win_rate = dfh['pnl'].gt(0).mean()*100
        if verbose:
            print(f"Bets: {len(dfh)}, Final Bank: £{bank:.2f}")
            print(f"Total P&L: £{total_pnl:.2f}, ROI: {roi:.2f}%, Win Rate: {win_rate:.1f}%")
            print("Detailed history:")
            print(dfh[['file','minute','state','pred','actual','conf','odds','stake','pnl','bank']].to_string(index=False))
        return dfh, bank, total_pnl, total_staked, roi, win_rate

def run_analysis(flat_stake, initial_bank):
    import pandas as pd
    thresholds = [i/100 for i in range(50,100,5)]
    kellys = [i/10 for i in range(0,11)]
    results = []
    for t in thresholds:
        for k in kellys:
            tester = SingleBetTester(t, flat_stake, initial_bank, k)
            dfh, bank, total_pnl, total_staked, roi, win_rate = tester.run(verbose=False)
            results.append({
                'threshold': t,
                'kelly': k,
                'final_bank': bank,
                'pnl': total_pnl,
                'roi': roi,
                'win_rate': win_rate
            })
    df = pd.DataFrame(results)
    df = df.sort_values(by='roi', ascending=False).head(10)
    print(df.to_string(index=False))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.50, help="Confidence threshold to bet")
    parser.add_argument('--flat-stake', type=float, default=1.0, help="Flat stake per bet")
    parser.add_argument('--initial-bank', type=float, default=100.0, help="Starting bank")
    parser.add_argument('--kelly', type=float, default=0.0, help="Fractional Kelly stake (0 for flat-stake mode)")
    parser.add_argument('--analysis', action='store_true',
                        help="Run grid analysis of confidence thresholds and Kelly fractions")
    args = parser.parse_args()
    if args.analysis:
        run_analysis(args.flat_stake, args.initial_bank)
    else:
        tester = SingleBetTester(args.threshold, args.flat_stake,
                                 args.initial_bank, args.kelly)
        tester.run()