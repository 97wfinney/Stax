import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- CONFIGURATION ---
TRAINING_DIR = Path("../data/Training")
BACKTEST_DIR = Path("../data/Backtest")
MODEL_OUTPUT_DIR = Path("./xgboost_model")
MOMENTUM_WINDOW = 5    # must match backtester
STAKE_PER_BET = 10.0
DEFAULT_STRATEGIES = [10, 20, 30, 45, 60]


def get_team_names(match_name: str, odds_keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if ' vs ' in match_name:
        teams = match_name.split(' vs ')
        return teams[0].strip(), teams[1].strip()
    non_draw = [k for k in odds_keys if k.lower() != 'draw']
    if len(non_draw) >= 2:
        return non_draw[0], non_draw[1]
    return None, None


def process_match_file(json_path: Path) -> Optional[pd.DataFrame]:
    with open(json_path) as f:
        data = json.load(f)
    if not data:
        return None
    match_name = data[0].get('match', 'Unknown')
    final_home, final_away = map(int, data[-1]['score'].split(' - '))
    if final_home > final_away:
        final_outcome = 0
    elif final_away > final_home:
        final_outcome = 1
    else:
        final_outcome = 2
    home, away = get_team_names(match_name, list(data[0]['odds'].values())[0].keys())
    if not home:
        return None
    rows = []
    for i, e in enumerate(data):
        h_score, a_score = map(int, e['score'].split(' - '))
        h_odds, a_odds, d_odds = [], [], []
        for bo in e['odds'].values():
            h_odds.append(bo.get(home))
            a_odds.append(bo.get(away))
            d_odds.append(bo.get('Draw'))
        h_odds = [o for o in h_odds if o is not None]
        a_odds = [o for o in a_odds if o is not None]
        d_odds = [o for o in d_odds if o is not None]
        rows.append({
            'time_elapsed_s': i*40,
            'home_score': h_score, 'away_score': a_score,
            'score_diff': h_score - a_score,
            'avg_home_odds': np.mean(h_odds) if h_odds else np.nan,
            'avg_away_odds': np.mean(a_odds) if a_odds else np.nan,
            'avg_draw_odds': np.mean(d_odds) if d_odds else np.nan,
            'std_home_odds': np.std(h_odds) if len(h_odds)>1 else 0,
            'std_away_odds': np.std(a_odds) if len(a_odds)>1 else 0,
            'std_draw_odds': np.std(d_odds) if len(d_odds)>1 else 0,
        })
    df = pd.DataFrame(rows)
    df['home_odds_momentum'] = df['avg_home_odds'].diff().rolling(MOMENTUM_WINDOW).mean()
    df['away_odds_momentum'] = df['avg_away_odds'].diff().rolling(MOMENTUM_WINDOW).mean()
    df['draw_odds_momentum'] = df['avg_draw_odds'].diff().rolling(MOMENTUM_WINDOW).mean()
    df['prob_home'] = 1/df['avg_home_odds']
    df['prob_away'] = 1/df['avg_away_odds']
    df['prob_draw'] = 1/df['avg_draw_odds']
    df['final_outcome'] = final_outcome
    return df


def create_training_data(training_dir: Path) -> pd.DataFrame:
    files = list(training_dir.glob('*.json'))
    dfs = []
    for f in files:
        df = process_match_file(f)
        if df is not None:
            dfs.append(df)
    master = pd.concat(dfs, ignore_index=True)
    master.dropna(inplace=True)
    return master


def train_model(df: pd.DataFrame) -> Tuple[xgb.XGBClassifier, StandardScaler]:
    features = [c for c in df.columns if c!='final_outcome']
    X, y = df[features], df['final_outcome']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train)
    X_val_np   = scaler.transform(X_val)
    # wrap into DataFrame so model picks up column names
    X_train_scaled = pd.DataFrame(X_train_np, columns=features, index=X_train.index)
    X_val_scaled   = pd.DataFrame(X_val_np,   columns=features, index=X_val.index)
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=1000, learning_rate=0.05,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        early_stopping_rounds=50, random_state=42
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=100)
    preds = model.predict(X_val_scaled)
    print(f"Validation accuracy: {accuracy_score(y_val,preds):.4f}")
    print(classification_report(y_val, preds,
          target_names=['Home','Away','Draw']))
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_DIR/"xgboost_model.joblib")
    joblib.dump(scaler, MODEL_OUTPUT_DIR/"feature_scaler.joblib")
    print(f"Saved model+scaler to {MODEL_OUTPUT_DIR}")
    return model, scaler


class XGBoostProfitabilityBacktester:
    def __init__(self, model_dir: Path, backtest_dir: Path):
        self.model = joblib.load(model_dir/"xgboost_model.joblib")
        self.scaler = joblib.load(model_dir/"feature_scaler.joblib")
        self.model_features = [f for f in self.model.get_booster().feature_names if f is not None]
        self.backtest_dir = backtest_dir
        self.results = []
    def _engineer(self, path: Path):
        return process_match_file(path)
    def run(self, strategies: List[int]):
        files = list(self.backtest_dir.glob('*.json'))
        for file in files:
            df = self._engineer(file)
            if df is None: continue
            actual = df['final_outcome'].iloc[0]
            for m in strategies:
                t = m*60
                row = df.iloc[(df['time_elapsed_s']-t).abs().idxmin()]
                if row.isnull().any(): continue
                feats = row[self.model_features]
                scaled = self.scaler.transform(feats.values.reshape(1,-1))
                probs = self.model.predict_proba(scaled)[0]
                pred = np.argmax(probs)
                odds_map = {0:'avg_home_odds',1:'avg_away_odds',2:'avg_draw_odds'}
                bet_odds = row[odds_map[pred]]
                pnl = (STAKE_PER_BET*bet_odds - STAKE_PER_BET) if pred==actual else -STAKE_PER_BET
                self.results.append({
                    'strategy':m,'pnl':pnl,'correct':pred==actual
                })
        # summary
        dfr = pd.DataFrame(self.results)
        summary = dfr.groupby('strategy').agg(
            total_bets=('pnl','size'), total_pl=('pnl','sum'),
            win_rate=('correct',lambda x: x.mean()*100)
        )
        summary['roi'] = summary['total_pl']/(summary['total_bets']*STAKE_PER_BET)*100
        print(summary.round(2))
        # plot
        fig,ax1=plt.subplots()
        sns.barplot(x=summary.index,y='total_pl',data=summary,ax=ax1)
        ax1.set_ylabel('Total P&L (£)')
        ax2=ax1.twinx()
        sns.lineplot(x=summary.index,y='roi',data=summary,ax=ax2,marker='o')
        ax2.set_ylabel('ROI (%)')
        plt.show()
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','backtest','all'], default='all')
    parser.add_argument('--strategies', nargs='+', type=int, default=DEFAULT_STRATEGIES)
    args = parser.parse_args()
    try:
        if args.mode in ('train','all'):
            print('Starting training...')
            master = create_training_data(TRAINING_DIR)
            train_model(master)
        if args.mode in ('backtest','all'):
            print('Starting backtest...')
            bt = XGBoostProfitabilityBacktester(model_dir=MODEL_OUTPUT_DIR, backtest_dir=BACKTEST_DIR)
            bt.run(args.strategies)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()