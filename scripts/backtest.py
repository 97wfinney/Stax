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
from tqdm import tqdm

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

def calculate_match_minute(time_elapsed_s):
    """Match the training time calculation exactly."""
    if time_elapsed_s <= 45 * 60:
        return int(time_elapsed_s / 60)
    else:
        return int((time_elapsed_s - HALF_TIME_BREAK) / 60)

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
    def __init__(self, threshold, flat_stake, initial_bank, kelly_fraction, real_mode=False):
        self.threshold = threshold
        self.flat_stake = flat_stake
        self.initial_bank = initial_bank
        self.kelly_fraction = kelly_fraction
        self.real_mode = real_mode
        self.cashed_out_total = 0.0
        self._load_models()

    def _load_models(self):
        # Try loading config from stax3, otherwise fallback to stax_kfold
        config_path = STAX_MODEL_DIR / 'stax_config.joblib'
        if not config_path.exists():
            config_path = STAX_MODEL_DIR.parent / 'stax_kfold' / 'stax_config.joblib'
        cfg = joblib.load(config_path)
        # load saved embedding mappings
        self.team_to_idx = cfg.get('team_to_idx', {})
        self.league_to_idx = cfg.get('league_to_idx', {})
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
        # load Keras neural meta-model (prefer simple_nn, fallback to nn)
        meta_model_path = STAX_MODEL_DIR / 'meta_model_simple_nn.h5'
        if not meta_model_path.exists():
            meta_model_path = STAX_MODEL_DIR / 'meta_model_nn.h5'
        self.meta_model = tf.keras.models.load_model(str(meta_model_path))
        self.meta_scaler = joblib.load(STAX_MODEL_DIR / 'meta_scaler_standard.joblib')

    def _predict_probs(self, df, data):
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
        # assemble full meta-features in exact order as training
        meta_features_list = []
        for i in range(df.shape[0]):
            # base model probabilities for this row
            p_lr_H, p_lr_A, p_lr_D = lr_p[i]
            p_xgb_H, p_xgb_A, p_xgb_D = xgb_p[i]
            p_lstm_H, p_lstm_A, p_lstm_D = lstm_p[i]

            # averages
            avg_H = (p_lr_H + p_xgb_H + p_lstm_H) / 3
            avg_A = (p_lr_A + p_xgb_A + p_lstm_A) / 3
            avg_D = (p_lr_D + p_xgb_D + p_lstm_D) / 3

            # disagreement metrics
            preds = np.array([
                [p_lr_H, p_lr_A, p_lr_D],
                [p_xgb_H, p_xgb_A, p_xgb_D],
                [p_lstm_H, p_lstm_A, p_lstm_D]
            ])
            avg_pred = preds.mean(axis=0)
            entropy = -np.sum(avg_pred * np.log(avg_pred + 1e-10))
            std_dev = preds.std(axis=0).mean()
            max_diff = 0
            for j in range(3):
                for k in range(j+1, 3):
                    diff = np.abs(preds[j] - preds[k]).max()
                    if diff > max_diff:
                        max_diff = diff

            # other features
            time_factor = min(i / 90, 1.0)
            odds_vals = df[['avg_home_odds','avg_away_odds','avg_draw_odds']].iloc[i].values
            market_overround = (1 / odds_vals).sum() - 1
            score_diff = df['score_diff'].iloc[i]

            # compose feature vector in training order
            row_feats = [
                p_lr_H, p_lr_A, p_lr_D,
                p_xgb_H, p_xgb_A, p_xgb_D,
                p_lstm_H, p_lstm_A, p_lstm_D,
                avg_H, avg_A, avg_D,
                entropy, std_dev, max_diff,
                time_factor, market_overround, score_diff
            ]
            meta_features_list.append(row_feats)

        meta_features = np.array(meta_features_list)
        # scale meta-features
        scaled_meta = self.meta_scaler.transform(meta_features)
        # prepare embedding indices using saved mappings
        n = df.shape[0]
        # extract match metadata
        home_name, away_name = data[0]['match'].split(' vs ')
        league_name = data[0].get('league', 'unknown')
        home_idx = self.team_to_idx.get(home_name, self.team_to_idx.get('unknown', 0))
        away_idx = self.team_to_idx.get(away_name, self.team_to_idx.get('unknown', 0))
        league_idx = self.league_to_idx.get(league_name, self.league_to_idx.get('unknown', 0))
        home_arr = np.full((n,), home_idx, dtype=int)
        away_arr = np.full((n,), away_idx, dtype=int)
        league_arr = np.full((n,), league_idx, dtype=int)
        # minute buckets: 0=[0-29],1=[30-59],2=[60-89],3=[90+]
        minutes = df['time_elapsed_s'].apply(calculate_match_minute).values
        minute_bucket = np.where(minutes < 30, 0,
                            np.where(minutes < 60, 1,
                                np.where(minutes < 90, 2, 3)))
        # score_diff buckets: <=-2=0, -2<..<=-1=1, -1<..<0=2, 0<=..<1=3, >=1=4
        scores = df['score_diff'].astype(int).values
        score_diff_bucket = np.where(scores <= -2, 0,
                                np.where(scores <= -1, 1,
                                    np.where(scores < 0, 2,
                                        np.where(scores < 1, 3, 4))))
        # predict based on model type (simple_nn vs full model)
        model_name = getattr(self.meta_model, 'name', '')
        if 'simple_nn' in model_name or len(self.meta_model.inputs) == 1:
            # simple model expects only scaled features
            ens = self.meta_model.predict(scaled_meta, verbose=0)
        else:
            # full model expects embeddings and buckets
            ens = self.meta_model.predict([
                scaled_meta,
                home_arr,
                away_arr,
                league_arr,
                minute_bucket,
                score_diff_bucket
            ], verbose=0)
        return ens


    def run(self, verbose=True):
        bank = self.initial_bank
        history = []
        for f in tqdm(sorted(BACKTEST_DIR.glob('*.json')), desc="Backtesting matches"):
            data = json.load(open(f))
            df, outcome = get_match_features(data)
            if df is None:
                continue
            probs = self._predict_probs(df, data)
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
                # Cash out in real mode when bank hits £1000
                if self.real_mode and bank >= 2000.0:
                    # take everything above £100 and reset
                    self.cashed_out_total += bank - 100.0
                    bank = 100.0
                minute = calculate_match_minute(row.time_elapsed_s)
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
            print("\nBet size and bank progression:")
            # add result column to indicate win or loss
            dfh['result'] = dfh['pnl'].apply(lambda x: 'W' if x > 0 else 'L')
            # format stake and bank with commas and two decimal places
            df_formatted = dfh[['stake', 'result', 'bank']].copy()
            df_formatted['stake'] = df_formatted['stake'].map(lambda x: f"{x:,.2f}")
            df_formatted['bank'] = df_formatted['bank'].map(lambda x: f"{x:,.2f}")
            print(df_formatted.to_string(index=False))
            if self.real_mode:
                print(f"\nTotal cashed out: £{self.cashed_out_total:.2f}")
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
    parser.add_argument('--real',
                        action='store_true',
                        help="Real mode: cash out when bank reaches £1000 and reset bank to 100")
    args = parser.parse_args()
    if args.analysis:
        run_analysis(args.flat_stake, args.initial_bank)
    else:
        tester = SingleBetTester(
            args.threshold,
            args.flat_stake,
            args.initial_bank,
            args.kelly,
            args.real
        )
        tester.run()