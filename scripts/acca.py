#!/usr/bin/env python3
"""
Stax Meta-Model - ACCUMULATOR BACKTESTER (with Halftime Logic & HTML Report)

This script loads pre-trained models and runs a specific backtest on multiple
hardcoded sets of concurrent matches to evaluate accumulator performance.
It generates a detailed HTML report of the results.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STAX_MODEL_DIR = MODELS_DIR / "stax_kfold" # Use the K-Fold trained models
REPORT_FILE = Path(__file__).resolve().parent / "acca_backtest_report.html"

# --- Backtest Parameters ---
# Using a dictionary to hold all groups of concurrent matches for testing
ACCA_GROUPS = {
    "Group 1: 5-Fold EFL Acca (2025-02-22)": [
        "coventry_city_vs_preston_north_end__efl_match_data_2025-02-22_14-59-01.json",
        "norwich_city_vs_stoke_city__efl_match_data_2025-02-22_14-59-01.json",
        "portsmouth_vs_queens_park_rangers__efl_match_data_2025-02-22_14-59-01.json",
        "swansea_city_vs_blackburn_rovers__efl_match_data_2025-02-22_14-59-01.json",
        "west_bromwich_albion_vs_oxford_united__efl_match_data_2025-02-22_14-59-01.json"
    ],
    "Group 2: 11-Fold EFL Acca (2025-04-21)": [
        "cardiff_city_vs_oxford_united__efl_match_data_2025-04-21_14-59-01.json",
        "hull_city_vs_preston_north_end__efl_match_data_2025-04-21_14-59-01.json",
        "leeds_united_vs_stoke_city__efl_match_data_2025-04-21_14-59-01.json",
        "luton_vs_bristol_city__efl_match_data_2025-04-21_14-59-01.json",
        "millwall_vs_norwich_city__efl_match_data_2025-04-21_14-59-01.json",
        "plymouth_argyle_vs_coventry_city__efl_match_data_2025-04-21_14-59-01.json",
        "portsmouth_vs_watford__efl_match_data_2025-04-21_14-59-01.json",
        "queens_park_rangers_vs_swansea_city__efl_match_data_2025-04-21_14-59-01.json",
        "sheffield_wednesday_vs_middlesbrough__efl_match_data_2025-04-21_14-59-01.json",
        "sunderland_vs_blackburn_rovers__efl_match_data_2025-04-21_14-59-01.json",
        "west_bromwich_albion_vs_derby_county__efl_match_data_2025-04-21_14-59-01.json"
    ],
    "Group 3: 3-Fold EPL Acca (2025-04-20)": [
        "fulham_vs_chelsea__epl_match_data_2025-04-20_13-58-02.json",
        "ipswich_town_vs_arsenal__epl_match_data_2025-04-20_13-58-02.json",
        "manchester_united_vs_wolverhampton_wanderers__epl_match_data_2025-04-20_13-58-02.json"
    ],
    "Group 4: 4-Fold EPL Acca (2025-04-19)": [
        "brentford_vs_brighton_and_hove_albion__epl_match_data_2025-04-19_14-58-01.json",
        "crystal_palace_vs_bournemouth__epl_match_data_2025-04-19_14-58-01.json",
        "everton_vs_manchester_city__epl_match_data_2025-04-19_14-58-01.json",
        "west_ham_united_vs_southampton__epl_match_data_2025-04-19_14-58-01.json"
    ]
}

# These now represent ELAPSED time from kick-off in minutes.
ACCA_TIMES = [10, 15, 20, 25, 30, 35, 45, 52, 75, 90] # 52m = mid-halftime, 75m = 60 mins play, 90m = 75 mins play
ACCA_STAKE = 10.0
SEQUENCE_LENGTH = 5

# Feature lists
LR_FEATURES = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s', 'home_score', 'away_score', 'score_diff', 'avg_home_odds',
    'avg_away_odds', 'avg_draw_odds', 'std_home_odds', 'std_away_odds',
    'std_draw_odds', 'home_odds_momentum', 'away_odds_momentum',
    'draw_odds_momentum', 'prob_home', 'prob_away', 'prob_draw'
]


class StaxModelLoader:
    """A class dedicated to loading all necessary models and providing predictions."""
    
    def __init__(self):
        self.output_dir = STAX_MODEL_DIR
        self.models = {}
        self.scalers = {}
        self.meta_models = {}
        self.meta_scalers = {}
        self.model_weights = None
        self.ensemble_method = None

    def load_saved_models(self):
        """Load all previously saved base and meta models."""
        print("Loading saved models...")
        config_path = self.output_dir / "stax_config.joblib"
        if not config_path.exists(): raise FileNotFoundError(f"Config file not found at {config_path}")
        config = joblib.load(config_path)
        self.model_weights, self.ensemble_method = config['model_weights'], config['ensemble_method']
        
        # --- FIXED MODEL LOADING LOGIC ---
        lr_dir = self.output_dir / "logistic_regression"
        xgb_dir = self.output_dir / "xgboost"
        lstm_dir = self.output_dir / "lstm"

        self.models['lr'] = joblib.load(lr_dir / "model.joblib")
        self.scalers['lr'] = joblib.load(lr_dir / "scaler.joblib")
        
        self.models['xgb'] = joblib.load(xgb_dir / "model.joblib")
        self.scalers['xgb'] = joblib.load(xgb_dir / "scaler.joblib")
        
        self.models['lstm'] = tf.keras.models.load_model(lstm_dir / "model.h5")
        self.scalers['lstm'] = joblib.load(lstm_dir / "scaler.pkl")
        # --- END OF FIX ---

        for model_type in ['lr', 'rf', 'weighted_rf']:
            model_path = self.output_dir / f"meta_model_{model_type}.joblib"
            if model_path.exists(): self.meta_models[model_type] = joblib.load(model_path)
        
        self.meta_scalers['standard'] = joblib.load(self.output_dir / "meta_scaler_standard.joblib")
        print("All models loaded successfully!")

    def get_prediction_for_timestep(self, match_data, elapsed_time_in_minutes):
        """Generates a prediction for a single match at a specific elapsed time."""
        df_xgb, outcome = self.process_match_for_xgb(match_data)
        lstm_features_df = self.get_lstm_features_df(match_data)
        
        if df_xgb is None or df_xgb.empty: return None

        target_idx = int(elapsed_time_in_minutes * 60 / 40)
        
        if target_idx >= len(df_xgb):
            print(f"Warning: Time {elapsed_time_in_minutes}m (index {target_idx}) is beyond match data length ({len(df_xgb)}).")
            return None
            
        if target_idx < SEQUENCE_LENGTH: return None

        meta_features = self.get_meta_features_for_row(df_xgb, lstm_features_df, target_idx)
        if meta_features is None: return None

        probs = self.predict_with_best_model(meta_features)
        
        prediction_index = np.argmax(probs)
        prediction_odds = df_xgb.iloc[target_idx][['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].values[prediction_index]
        prediction_map = {0: "Home Win", 1: "Away Win", 2: "Draw"}
        
        return {
            "prediction_text": prediction_map[prediction_index],
            "prediction_index": prediction_index,
            "odds": prediction_odds,
            "actual_outcome": outcome
        }

    # --- Helper and data processing functions ---
    def create_weighted_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model_weights is None: raise ValueError("Model weights not loaded.")
        X_weighted = X.copy()
        X_weighted['weighted_H'] = (self.model_weights[0]*X['p_lr_H'] + self.model_weights[1]*X['p_xgb_H'] + self.model_weights[2]*X['p_lstm_H'])
        X_weighted['weighted_A'] = (self.model_weights[0]*X['p_lr_A'] + self.model_weights[1]*X['p_xgb_A'] + self.model_weights[2]*X['p_lstm_A'])
        X_weighted['weighted_D'] = (self.model_weights[0]*X['p_lr_D'] + self.model_weights[1]*X['p_xgb_D'] + self.model_weights[2]*X['p_lstm_D'])
        return X_weighted

    def get_team_names(self, match_name: str, odds_data: dict) -> Tuple[Optional[str], Optional[str]]:
        if ' vs ' in match_name: return match_name.split(' vs ')[0].strip(), match_name.split(' vs ')[1].strip()
        keys = [k for o in odds_data.values() for k in o if k.lower()!='draw']
        return (keys[0], keys[1]) if len(keys) >= 2 else (None, None)

    def process_match_for_lr(self, json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
        if not json_data: return None, -1
        name, odds = json_data[0].get('match', 'N/A'), json_data[0].get('odds', {})
        h_team, a_team = self.get_team_names(name, odds)
        if not h_team: return None, -1
        score = json_data[-1]['score'].split(' - ')
        outcome = 0 if int(score[0]) > int(score[1]) else 1 if int(score[1]) > int(score[0]) else 2
        rows = [{'home_score': int(e['score'].split(' - ')[0]), 'away_score': int(e['score'].split(' - ')[1]),
                 'avg_home_odds': np.mean([o.get(h_team) for o in e.get('odds', {}).values() if o.get(h_team)]),
                 'avg_away_odds': np.mean([o.get(a_team) for o in e.get('odds', {}).values() if o.get(a_team)]),
                 'avg_draw_odds': np.mean([o.get('Draw') for o in e.get('odds', {}).values() if o.get('Draw')])} for e in json_data]
        return pd.DataFrame(rows).dropna(), outcome

    def process_match_for_xgb(self, json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
        df, outcome = self.process_match_for_lr(json_data)
        if df is None or df.empty: return None, -1
        df['score_diff'] = df['home_score'] - df['away_score']
        for col in ['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']:
            df[f'std_{col.split("_")[1]}_odds'] = df[col].rolling(5).std().fillna(0)
            df[f'{col.split("_")[1]}_odds_momentum'] = df[col].diff().rolling(5).mean()
        df['prob_home'] = 1 / df['avg_home_odds']; df['prob_away'] = 1 / df['avg_away_odds']; df['prob_draw'] = 1 / df['avg_draw_odds']
        df['time_elapsed_s'] = df.index * 40
        return df.dropna(), outcome

    def get_lstm_features_df(self, json_data: List[Dict]) -> Optional[pd.DataFrame]:
        df, _ = self.process_match_for_lr(json_data)
        if df is None or df.empty: return None
        lstm_df = df[['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].copy()
        lstm_df['score_diff'] = df['home_score'] - df['away_score']
        return lstm_df

    def calculate_model_disagreement(self, pred_lr, pred_xgb, pred_lstm):
        preds = np.array([pred_lr, pred_xgb, pred_lstm])
        avg_pred = preds.mean(axis=0)
        entropy = -np.sum(avg_pred * np.log(avg_pred + 1e-9))
        std_dev = preds.std(axis=0).mean()
        max_diff = np.abs(preds[:, np.newaxis, :] - preds[np.newaxis, :, :]).max()
        return entropy, std_dev, max_diff

    def get_meta_features_for_row(self, df_xgb, lstm_features_df, target_idx):
        try:
            target_row = df_xgb.iloc[target_idx]
            lr_feats = self.scalers['lr'].transform(target_row[LR_FEATURES].values.reshape(1, -1))
            pred_lr = self.models['lr'].predict_proba(lr_feats)[0]
            xgb_feats = self.scalers['xgb'].transform(target_row[XGB_FEATURES].values.reshape(1, -1))
            pred_xgb = self.models['xgb'].predict_proba(xgb_feats)[0]
            seq_df = lstm_features_df.iloc[target_idx - SEQUENCE_LENGTH + 1 : target_idx + 1]
            if len(seq_df) != SEQUENCE_LENGTH: return None
            lstm_feats = self.scalers['lstm'].transform(seq_df.values).reshape(1, SEQUENCE_LENGTH, -1)
            pred_lstm = self.models['lstm'].predict(lstm_feats, verbose=0)[0]
            entropy, std_dev, max_diff = self.calculate_model_disagreement(pred_lr, pred_xgb, pred_lstm)
            return pd.DataFrame([{'p_lr_H': pred_lr[0], 'p_lr_A': pred_lr[1], 'p_lr_D': pred_lr[2],
                'p_xgb_H': pred_xgb[0], 'p_xgb_A': pred_xgb[1], 'p_xgb_D': pred_xgb[2],
                'p_lstm_H': pred_lstm[0], 'p_lstm_A': pred_lstm[1], 'p_lstm_D': pred_lstm[2],
                'avg_H': np.mean([pred_lr[0], pred_xgb[0], pred_lstm[0]]), 'avg_A': np.mean([pred_lr[1], pred_xgb[1], pred_lstm[1]]), 'avg_D': np.mean([pred_lr[2], pred_xgb[2], pred_lstm[2]]),
                'entropy': entropy, 'std_dev': std_dev, 'max_disagreement': max_diff,
                'time_factor': min(target_idx/90, 1.0), 'market_overround': (1/target_row['avg_home_odds']+1/target_row['avg_away_odds']+1/target_row['avg_draw_odds'])-1,
                'score_diff': target_row['score_diff']}])
        except Exception:
            return None

    def predict_with_best_model(self, meta_features: pd.DataFrame) -> np.ndarray:
        method = self.ensemble_method
        if method == 'lr': return self.meta_models['lr'].predict_proba(self.meta_scalers['standard'].transform(meta_features))[0]
        if method == 'rf': return self.meta_models['rf'].predict_proba(meta_features)[0]
        if method == 'weighted_rf': return self.meta_models['weighted_rf'].predict_proba(self.create_weighted_features(meta_features))[0]
        weights = self.model_weights or [0.33, 0.34, 0.33]
        pred = (weights[0] * meta_features[['p_lr_H', 'p_lr_A', 'p_lr_D']].values +
                weights[1] * meta_features[['p_xgb_H', 'p_xgb_A', 'p_xgb_D']].values +
                weights[2] * meta_features[['p_lstm_H', 'p_lstm_A', 'p_lstm_D']].values)
        return pred[0] / pred[0].sum()


def run_acca_backtest(stax_model, match_files, time_intervals):
    """Runs the accumulator backtest for the specified matches and times."""
    
    html_output = ""
    all_match_data = {}
    print("Loading match data...")
    for file_name in match_files:
        file_path = DATA_DIR / "Backtest" / file_name
        if not file_path.exists():
            print(f"Error: Could not find {file_path}")
            html_output += f"<p>Error: Could not find {file_path}</p>"
            return html_output
        with open(file_path, 'r') as f:
            all_match_data[file_name] = json.load(f)
    print("Match data loaded.\n")
    
    for time_min in time_intervals:
        terminal_string = f"--- Testing Accumulator at {time_min} Minutes (Elapsed Time) ---"
        html_string = f"<h2>Testing Accumulator at {time_min} Minutes (Elapsed Time)</h2>"
        print(terminal_string)

        acca_legs, is_valid_acca = [], True
        
        for file_name, match_data in all_match_data.items():
            match_name_short = file_name.split('__epl_')[0].split('__efl_')[0].replace('_', ' ').title()
            prediction_info = stax_model.get_prediction_for_timestep(match_data, time_min)
            
            if prediction_info is None:
                msg = f"  - Could not generate prediction for {match_name_short}. Skipping acca."
                print(msg)
                html_string += f"<p class='error'>{msg}</p>"
                is_valid_acca = False
                break
            
            prediction_info['match_name'] = match_name_short
            acca_legs.append(prediction_info)
        
        if not is_valid_acca:
            terminal_string += "\n----------------------------------------\n"
            html_string += "<hr>"
            html_output += html_string
            print("-" * 40 + "\n")
            continue
            
        total_odds, all_legs_correct = 1.0, True
        
        stake_str = f"  Stake: £{ACCA_STAKE:.2f}"
        legs_header_str = "  Accumulator Legs:"
        print(stake_str); print(legs_header_str)
        
        html_string += f"<p><b>Stake:</b> £{ACCA_STAKE:.2f}</p><h4>Accumulator Legs:</h4><table>"
        html_string += "<tr><th>Match</th><th>Prediction</th><th>Odds</th><th>Result</th></tr>"

        for leg in acca_legs:
            total_odds *= leg['odds']
            is_correct = leg['prediction_index'] == leg['actual_outcome']
            if not is_correct: all_legs_correct = False
            
            result_icon = "✅" if is_correct else "❌"
            leg_str = f"    - {leg['match_name']:<45} -> Predicted: {leg['prediction_text']:<10} @ {leg['odds']:.2f} {result_icon}"
            print(leg_str)
            html_string += f"<tr><td>{leg['match_name']}</td><td>{leg['prediction_text']}</td><td>{leg['odds']:.2f}</td><td>{result_icon}</td></tr>"

        html_string += "</table>"
        potential_payout = ACCA_STAKE * total_odds
        pnl = (potential_payout - ACCA_STAKE) if all_legs_correct else -ACCA_STAKE
        
        result_str = f"  Result: {'🎉 WIN 🎉' if all_legs_correct else '😭 LOSS 😭'}"
        pnl_str = f"  Profit/Loss: £{pnl:.2f}"
        
        print(f"\n  Total Odds: {total_odds:.2f}")
        print(f"  Potential Payout: £{potential_payout:.2f}")
        print(result_str); print(pnl_str)
        print("-" * 40 + "\n")

        html_string += f"<div class='summary'><b>Total Odds:</b> {total_odds:.2f}<br/>"
        html_string += f"<b>Potential Payout:</b> £{potential_payout:.2f}<br/>"
        html_string += f"<b>Result: <span class='{'win' if all_legs_correct else 'loss'}'>{result_str.split(': ')[1]}</span></b><br/>"
        html_string += f"<b>Profit/Loss:</b> £{pnl:.2f}</div><hr>"
        
        html_output += html_string
        
    return html_output

def generate_html_report(all_html_content):
    """Generates the final HTML report file."""
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stax Accumulator Backtest Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f4f7f9; color: #333; }}
            .container {{ max-width: 900px; margin: 20px auto; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a2b4d; border-bottom: 2px solid #e1e8ed; padding-bottom: 10px; }}
            h2 {{ color: #2a4c8d; border-bottom: 1px solid #e1e8ed; padding-bottom: 8px; margin-top: 30px;}}
            h3 {{ color: #333; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .summary {{ margin-top: 15px; padding: 15px; background-color: #eef2f5; border-left: 4px solid #6c757d; }}
            .win {{ color: #28a745; font-weight: bold; }}
            .loss {{ color: #dc3545; font-weight: bold; }}
            .error {{ color: #dc3545; font-style: italic; }}
            hr {{ border: none; height: 1px; background-color: #e1e8ed; margin: 30px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Stax Accumulator Backtest Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {''.join(all_html_content)}
        </div>
    </body>
    </html>
    """
    with open(REPORT_FILE, "w") as f:
        f.write(html_template)
    print(f"\n✅ HTML report saved to: {REPORT_FILE}")

def main():
    """Main function to initialize models and run the backtests."""
    print("=== Stax Accumulator Backtester ===")
    
    stax_model = StaxModelLoader()
    stax_model.load_saved_models()
    
    all_results_html = []
    
    for group_name, match_files in ACCA_GROUPS.items():
        print(f"\n\n{'='*20}\nTesting: {group_name}\n{'='*20}")
        all_results_html.append(f"<h1>{group_name}</h1>")
        
        group_html = run_acca_backtest(stax_model, match_files, ACCA_TIMES)
        all_results_html.append(group_html)

    generate_html_report(all_results_html)
    print("\n--- All Accumulator Backtests Complete ---")

if __name__ == '__main__':
    main()
