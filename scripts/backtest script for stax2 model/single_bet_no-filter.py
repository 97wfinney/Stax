#!/usr/bin/env python3

"""

single_bet_v8.py



Final version with all bug fixes for styling and dependencies.

"""



import json

import argparse

from pathlib import Path

import pandas as pd

import numpy as np

import joblib

import tensorflow as tf

import matplotlib.pyplot as plt

import base64

from io import BytesIO



# --- HIDE HARMLESS WARNINGS ---

import warnings

# UserWarning is a built-in, no specific import is needed.

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



# --- ROBUST PATHING & CONSTANTS ---

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

MODELS_DIR = ROOT_DIR / "models"

BACKTEST_DIR = DATA_DIR / "Backtest"

STAX_MODEL_DIR = MODELS_DIR / "stax_kfold"



# Feature lists

LR_FEATURES = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']

XGB_FEATURES = [

    'time_elapsed_s', 'home_score', 'away_score', 'score_diff',

    'avg_home_odds', 'avg_away_odds', 'avg_draw_odds',

    'std_home_odds', 'std_away_odds', 'std_draw_odds',

    'home_odds_momentum', 'away_odds_momentum', 'draw_odds_momentum',

    'prob_home', 'prob_away', 'prob_draw'

]

LSTM_FEATURES = ['avg_home_odds', 'avg_away_odds', 'avg_draw_odds', 'score_diff']

SEQUENCE_LENGTH = 5



# --- DATA PROCESSING ---

def get_match_features(data):

    if not data: return None, -1

    home_team, away_team = data[0]['match'].split(' vs ')

    fh, fa = map(int, data[-1]['score'].split(' - '))

    outcome = 0 if fh > fa else (1 if fa > fh else 2)

    rows = []

    for i, e in enumerate(data):

        hs, aws = map(int, e['score'].split(' - '))

        odds = e['odds']

        h_odds = [o.get(home_team) for o in odds.values() if o.get(home_team)]

        a_odds = [o.get(away_team) for o in odds.values() if o.get(away_team)]

        d_odds = [o.get('Draw') for o in odds.values() if o.get('Draw')]

        rows.append({

            'time_elapsed_s': i * 40, 'home_score': hs, 'away_score': aws,

            'avg_home_odds': np.mean(h_odds) if h_odds else np.nan,

            'avg_away_odds': np.mean(a_odds) if a_odds else np.nan,

            'avg_draw_odds': np.mean(d_odds) if d_odds else np.nan,

        })

    df = pd.DataFrame(rows).dropna()

    if df.empty: return None, -1

    df['score_diff'] = df['home_score'] - df['away_score']

    for col in ['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']:

        df[f'std_{col.split("_")[1]}_odds'] = df[col].rolling(5).std().fillna(0)

        df[f'{col.split("_")[1]}_odds_momentum'] = df[col].diff().rolling(5).mean().fillna(0)

    df['prob_home'] = 1 / df['avg_home_odds']

    df['prob_away'] = 1 / df['avg_away_odds']

    df['prob_draw'] = 1 / df['avg_draw_odds']

    return df.dropna(), outcome



# --- ANALYZER WITH BANKROLL ---

class SingleBetHTMLAnalyzer:

    def __init__(self, models_dir, threshold, kelly_fraction, initial_bank):

        self.threshold = threshold

        self.kelly_fraction = kelly_fraction

        self.initial_bank = initial_bank

        self._load_models(models_dir)



    def _load_models(self, md):

        try:

            cfg = joblib.load(md / 'stax_config.joblib')

            self.weights = cfg.get('model_weights', [0.34, 0.33, 0.33])

            self.models, self.scalers = {}, {}

            for model_name in ['lr', 'xgb', 'lstm']:

                p = md / model_name.replace('xgb', 'xgboost').replace('lr', 'logistic_regression')

                if model_name == 'lstm':

                    self.models['lstm'] = tf.keras.models.load_model(str(p / 'model.h5'))

                    self.scalers['lstm'] = joblib.load(p / 'scaler.pkl')

                else:

                    self.models[model_name] = joblib.load(p / 'model.joblib')

                    self.scalers[model_name] = joblib.load(p / 'scaler.joblib')

            print("All models loaded successfully.")

        except FileNotFoundError as e:

            print(f"Error loading models: {e}. Please check your project's directory structure.")

            exit(1)



    def _get_predictions(self, features_df, lstm_features_df):

        lr_feats = self.scalers['lr'].transform(features_df[LR_FEATURES])

        xgb_feats = self.scalers['xgb'].transform(features_df[XGB_FEATURES])

        sequences = []

        for i in range(len(lstm_features_df)):

            seq = lstm_features_df.iloc[max(0, i - SEQUENCE_LENGTH + 1):i+1].values

            if len(seq) < SEQUENCE_LENGTH:

                seq = np.vstack([np.zeros((SEQUENCE_LENGTH - len(seq), seq.shape[1])), seq])

            sequences.append(seq)

        sequences = np.array(sequences)

        scaled_seqs = self.scalers['lstm'].transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape)

        p_lr = self.models['lr'].predict_proba(lr_feats)

        p_xgb = self.models['xgb'].predict_proba(xgb_feats)

        p_lstm = self.models['lstm'].predict(scaled_seqs, verbose=0)

        return p_lr, p_xgb, p_lstm



    def _kelly_stake(self, probability, odds, bank):

        edge = probability * odds - 1

        if edge <= 0: return 0.0

        fraction = edge / (odds - 1)

        return min(fraction * bank * self.kelly_fraction, bank)



    def analyse_match(self, path, current_bank):

        try:

            with open(path, 'r') as f: data = json.load(f)

        except (json.JSONDecodeError, FileNotFoundError): return None, current_bank

        features_df, final_outcome = get_match_features(data)

        if features_df is None or features_df.empty: return None, current_bank

        p_lr, p_xgb, p_lstm = self._get_predictions(features_df, features_df[LSTM_FEATURES])

        ensemble_probs = (self.weights[0] * p_lr + self.weights[1] * p_xgb + self.weights[2] * p_lstm)



        for i, row in features_df.iterrows():

            if i < SEQUENCE_LENGTH: continue

            confidence = ensemble_probs[i].max()

            if confidence >= self.threshold:

                pred_idx = np.argmax(ensemble_probs[i])

                odds_cols = ['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']

                selected_odds = row[odds_cols[pred_idx]]

                stake = self._kelly_stake(confidence, selected_odds, current_bank)

                if stake <= 0.01: continue

                pnl = (stake * selected_odds - stake) if pred_idx == final_outcome else -stake

                return {

                    'file': path.name, 'minute': int(row['time_elapsed_s'] // 60),

                    'score_state': f"{int(row['home_score'])}-{int(row['away_score'])}",

                    'prediction': ['Home', 'Away', 'Draw'][pred_idx],

                    'actual': ['Home', 'Away', 'Draw'][final_outcome],

                    'confidence': confidence, 'odds': selected_odds,

                    'stake': stake, 'pnl': pnl, 'bank': current_bank + pnl

                }, current_bank + pnl

        return None, current_bank



    def run(self, backtest_dir):

        bank = self.initial_bank

        bet_history = []

        files = sorted(Path(backtest_dir).glob('*.json'))

        print(f"Starting backtest with initial bank: £{bank:.2f}")

        print(f"Found {len(files)} match files in {backtest_dir}.")

        for file in files:

            bet_result, bank = self.analyse_match(file, bank)

            if bet_result: bet_history.append(bet_result)

        if not bet_history:

            print("No bets were placed during the backtest.")

            return

        self.generate_html_report(pd.DataFrame(bet_history))



    def _calculate_max_drawdown(self, bank_progression):

        peak = -np.inf

        max_drawdown = 0

        for value in bank_progression:

            if value > peak: peak = value

            drawdown = (peak - value) / peak if peak > 0 else 0

            if drawdown > max_drawdown: max_drawdown = drawdown

        return max_drawdown



    def _style_pnl(self, val):

        color = '#d62728' if val < 0 else '#2ca02c' if val > 0 else '#333'

        return f'color: {color}; font-weight: bold;'



    def _plot_to_base64(self, fig):

        buf = BytesIO()

        fig.savefig(buf, format='png', bbox_inches='tight')

        plt.close(fig)

        return base64.b64encode(buf.getvalue()).decode('utf-8')



    def generate_html_report(self, df):

        # --- 1. Calculate Metrics ---

        bank_progression = [self.initial_bank] + df['bank'].tolist()

        final_bank = bank_progression[-1]

        total_pnl = df['pnl'].sum()

        total_staked = df['stake'].sum()

        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

        win_rate = df['pnl'].gt(0).mean() * 100

        num_bets = len(df)

        avg_stake = df['stake'].mean()

        max_drawdown = self._calculate_max_drawdown(bank_progression) * 100

        biggest_win = df['pnl'].max()

        biggest_loss = df['pnl'].min()



        # --- 2. Create Charts ---

        # Bankroll Chart

        fig_bank, ax_bank = plt.subplots(figsize=(10, 5))

        ax_bank.plot(bank_progression, marker='o', linestyle='-', markersize=4, color='#1f77b4')

        ax_bank.set_title('Bankroll Progression', fontsize=14)

        ax_bank.set_xlabel('Bet Number', fontsize=10)

        ax_bank.set_ylabel('Bank (£)', fontsize=10)

        ax_bank.grid(True, linestyle='--', alpha=0.6)

        chart_bank_b64 = self._plot_to_base64(fig_bank)



        # P&L by Prediction Type Chart

        pnl_by_pred = df.groupby('prediction')['pnl'].sum()

        fig_pnl, ax_pnl = plt.subplots(figsize=(6, 4))

        pnl_by_pred.plot(kind='bar', ax=ax_pnl, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

        ax_pnl.set_title('P&L by Prediction Type', fontsize=14)

        ax_pnl.set_ylabel('Total P&L (£)', fontsize=10)

        ax_pnl.tick_params(axis='x', rotation=0)

        ax_pnl.grid(axis='y', linestyle='--', alpha=0.6)

        chart_pnl_b64 = self._plot_to_base64(fig_pnl)



        # --- 3. Deep Dive Analysis ---

        # By Prediction

        pred_summary = df.groupby('prediction').agg(

            bets=('pnl', 'count'), win_rate=('pnl', lambda x: x.gt(0).mean() * 100),

            total_pnl=('pnl', 'sum'), roi=('pnl', lambda x: x.sum() / df.loc[x.index, 'stake'].sum() * 100 if df.loc[x.index, 'stake'].sum() > 0 else 0)

        ).reset_index()

        

        # By Time

        df['time_band'] = pd.cut(df['minute'], bins=[0, 30, 60, 120], labels=['0-30', '31-60', '61+'], right=False)

        time_summary = df.groupby('time_band', observed=True).agg(

            bets=('pnl', 'count'), win_rate=('pnl', lambda x: x.gt(0).mean() * 100),

            total_pnl=('pnl', 'sum'), roi=('pnl', lambda x: x.sum() / df.loc[x.index, 'stake'].sum() * 100 if df.loc[x.index, 'stake'].sum() > 0 else 0)

        ).reset_index()



        # By Odds

        df['odds_band'] = pd.cut(df['odds'], bins=[1, 1.5, 2.0, 3.0, 100], labels=['1.0-1.5', '1.51-2.0', '2.01-3.0', '3.01+'], right=False)

        odds_summary = df.groupby('odds_band', observed=True).agg(

            bets=('pnl', 'count'), win_rate=('pnl', lambda x: x.gt(0).mean() * 100),

            total_pnl=('pnl', 'sum'), roi=('pnl', lambda x: x.sum() / df.loc[x.index, 'stake'].sum() * 100 if df.loc[x.index, 'stake'].sum() > 0 else 0)

        ).reset_index()

        

        # --- 4. Render HTML ---

        # THIS IS THE CORRECTED LINE FOR THE DETAILED HISTORY TABLE

        df_styled = df.style.format({

            'confidence': '{:.2%}', 'odds': '{:.2f}', 'stake': '£{:.2f}', 'pnl': '£{:.2f}', 'bank': '£{:.2f}'

        }).map(self._style_pnl, subset=['pnl'])



        # THIS IS THE CORRECTED HELPER FUNCTION FOR THE SUMMARY TABLES

        def render_table(summary_df):

            return summary_df.style.format({

                'win_rate': '{:.2f}%', 'total_pnl': '£{:.2f}', 'roi': '{:.2f}%'

            }).map(self._style_pnl, subset=['total_pnl', 'roi']).to_html(index=False)



        html = f"""

        <html><head><title>Enhanced Backtest Report</title><style>

            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f4f7f6; color: #333; }}

            .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}

            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}

            .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }}

            .stat-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; border-left: 5px solid #3498db; }}

            .stat-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}

            .stat-label {{ color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; }}

            .positive {{ color: #27ae60 !important; font-weight: bold; }} .negative {{ color: #c0392b !important; font-weight: bold; }}

            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em;}}

            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}

            th {{ background-color: #34495e; color: white; }}

            tr:nth-child(even) {{ background-color: #f2f2f2; }}

            .chart-container {{ display: flex; gap: 20px; justify-content: center; align-items: flex-start; flex-wrap: wrap; margin: 20px 0; }}

        </style></head><body><div class="container">

            <h1>📈 Single Bet Backtest Report</h1>

            <h2>Overall Performance</h2>

            <div class="stat-grid">

                <div class="stat-card"><div class="stat-value">£{final_bank:.2f}</div><div class="stat-label">Final Bank</div></div>

                <div class="stat-card"><div class="stat-value {'positive' if total_pnl > 0 else 'negative'}">£{total_pnl:.2f}</div><div class="stat-label">Total P&L</div></div>

                <div class="stat-card"><div class="stat-value {'positive' if roi > 0 else 'negative'}">{roi:.2f}%</div><div class="stat-label">ROI</div></div>

                <div class="stat-card"><div class="stat-value">{win_rate:.2f}%</div><div class="stat-label">Win Rate</div></div>

                <div class="stat-card"><div class="stat-value negative">{max_drawdown:.2f}%</div><div class="stat-label">Max Drawdown</div></div>

                <div class="stat-card"><div class="stat-value">{num_bets}</div><div class="stat-label">Number of Bets</div></div>

            </div>

            <div class="stat-grid">

                <div class="stat-card"><div class="stat-value">£{avg_stake:.2f}</div><div class="stat-label">Avg. Stake</div></div>

                <div class="stat-card"><div class="stat-value positive">£{biggest_win:.2f}</div><div class="stat-label">Biggest Win</div></div>

                <div class="stat-card"><div class="stat-value negative">£{biggest_loss:.2f}</div><div class="stat-label">Biggest Loss</div></div>

            </div>

            <h2>Visual Analysis</h2>

            <div class="chart-container">

                <img src="data:image/png;base64,{chart_bank_b64}" alt="Bankroll Chart" style="max-width: 65%;">

                <img src="data:image/png;base64,{chart_pnl_b64}" alt="P&L Chart" style="max-width: 30%;">

            </div>

            <h2>Performance Deep Dive</h2>

            <h3>By Prediction Type</h3>{render_table(pred_summary)}

            <h3>By Time of Bet (Minute)</h3>{render_table(time_summary)}

            <h3>By Odds Range</h3>{render_table(odds_summary)}

            <h2>Detailed Bet History</h2>

            {df_styled.to_html(index=False)}

        </div></body></html>"""

        

        output_path = Path('single_bet_report_v2.html').resolve()

        with open(output_path, 'w', encoding='utf-8') as f: f.write(html)

        print(f"\n✅ Enhanced report written to: file://{output_path}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a single-bet-per-match backtest.")

    parser.add_argument('--threshold', type=float, default=0.50, help="Minimum confidence to place a bet.")

    parser.add_argument('--kelly-fraction', type=float, default=0.1, help="Fraction of the Kelly stake.")

    parser.add_argument('--initial-bank', type=float, default=100.0, help="Initial bankroll.")

    args = parser.parse_args()

    analyzer = SingleBetHTMLAnalyzer(STAX_MODEL_DIR, args.threshold, args.kelly_fraction, args.initial_bank)

    analyzer.run(BACKTEST_DIR)