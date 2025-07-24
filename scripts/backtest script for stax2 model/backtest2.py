#!/usr/bin/env python3
"""
Stax Meta-Model - EVENT-DRIVEN FLAT STAKE BACKTESTER (with Match Time Cutoff)
This script loads pre-trained models and runs a backtest that places one
flat £10 stake per match, at the first moment the betting conditions are met,
and only within a realistic match timeframe.
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
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
# This should point to the directory where your K-Fold models were saved
STAX_MODEL_DIR = MODELS_DIR / "stax_kfold" 

# Model parameters
STAKE_PER_BET = 10.0 # This is our flat stake
SEQUENCE_LENGTH = 5
# NEW: Add a cutoff to stop looking for bets after a realistic match time.
# 140 entries * 40 seconds/entry = 5600 seconds = ~93 minutes.
MAX_ITERATION_INDEX = 140 

# Feature lists
LR_FEATURES = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s', 'home_score', 'away_score', 'score_diff', 'avg_home_odds',
    'avg_away_odds', 'avg_draw_odds', 'std_home_odds', 'std_away_odds',
    'std_draw_odds', 'home_odds_momentum', 'away_odds_momentum',
    'draw_odds_momentum', 'prob_home', 'prob_away', 'prob_draw'
]


class EnhancedStaxModel:
    """A streamlined class to load models and provide predictions for backtesting."""
    
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
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        config = joblib.load(config_path)
        self.model_weights = config['model_weights']
        self.ensemble_method = config['ensemble_method']
        
        lr_dir = self.output_dir / "logistic_regression"
        xgb_dir = self.output_dir / "xgboost"
        lstm_dir = self.output_dir / "lstm"

        self.models['lr'] = joblib.load(lr_dir / "model.joblib")
        self.scalers['lr'] = joblib.load(lr_dir / "scaler.joblib")
        
        self.models['xgb'] = joblib.load(xgb_dir / "model.joblib")
        self.scalers['xgb'] = joblib.load(xgb_dir / "scaler.joblib")
        
        self.models['lstm'] = tf.keras.models.load_model(lstm_dir / "model.h5")
        self.scalers['lstm'] = joblib.load(lstm_dir / "scaler.pkl")
        
        for model_type in ['lr', 'rf', 'weighted_rf']:
            model_path = self.output_dir / f"meta_model_{model_type}.joblib"
            if model_path.exists():
                self.meta_models[model_type] = joblib.load(model_path)
        
        self.meta_scalers['standard'] = joblib.load(self.output_dir / "meta_scaler_standard.joblib")
        print("All models loaded successfully!")

    def create_weighted_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create weighted features using loaded weights."""
        if self.model_weights is None:
            raise ValueError("Model weights are not loaded.")
        
        X_weighted = X.copy()
        X_weighted['weighted_H'] = (self.model_weights[0] * X['p_lr_H'] + self.model_weights[1] * X['p_xgb_H'] + self.model_weights[2] * X['p_lstm_H'])
        X_weighted['weighted_A'] = (self.model_weights[0] * X['p_lr_A'] + self.model_weights[1] * X['p_xgb_A'] + self.model_weights[2] * X['p_lstm_A'])
        X_weighted['weighted_D'] = (self.model_weights[0] * X['p_lr_D'] + self.model_weights[1] * X['p_xgb_D'] + self.model_weights[2] * X['p_lstm_D'])
        return X_weighted

    # --- Data Processing functions ---
    def get_team_names(self, match_name: str, odds_data: dict) -> Tuple[Optional[str], Optional[str]]:
        if ' vs ' in match_name:
            teams = match_name.split(' vs ')
            return teams[0].strip(), teams[1].strip()
        non_draw_keys = [k for odds in odds_data.values() for k in odds if k.lower() != 'draw']
        if len(non_draw_keys) >= 2:
            return non_draw_keys[0], non_draw_keys[1]
        return None, None

    def process_match_for_lr(self, json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
        if not json_data: return None, -1
        match_name = json_data[0].get('match', 'Unknown')
        home_team, away_team = self.get_team_names(match_name, json_data[0].get('odds', {}))
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

    def process_match_for_xgb(self, json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
        df, final_outcome = self.process_match_for_lr(json_data)
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

    def get_lstm_features_df(self, json_data: List[Dict]) -> Optional[pd.DataFrame]:
        df, _ = self.process_match_for_lr(json_data)
        if df is None or df.empty: return None
        lstm_feats_df = df[['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].copy()
        lstm_feats_df['score_diff'] = df['home_score'] - df['away_score']
        return lstm_feats_df

    def calculate_model_disagreement(self, pred_lr, pred_xgb, pred_lstm):
        predictions = np.array([pred_lr, pred_xgb, pred_lstm])
        avg_pred = predictions.mean(axis=0)
        entropy = -np.sum(avg_pred * np.log(avg_pred + 1e-10))
        std_dev = predictions.std(axis=0).mean()
        max_diff = 0
        for i in range(3):
            for j in range(i + 1, 3):
                max_diff = max(max_diff, np.abs(predictions[i] - predictions[j]).max())
        return entropy, std_dev, max_diff


class ValueBettingBacktester:
    """Event-driven backtester using a flat stake."""
    
    def __init__(self, stax_model: EnhancedStaxModel):
        self.stax_model = stax_model
        self.results = []

    def should_place_bet(self, probs: np.ndarray, odds: List[float], 
                        confidence_threshold: float, value_threshold: float) -> Tuple[bool, int, float]:
        """Determine if a bet should be placed, returning a flat stake."""
        best_option = np.argmax(probs)
        confidence = probs[best_option]
        
        if confidence < confidence_threshold: return False, -1, 0
        
        expected_value = (probs[best_option] * odds[best_option]) - 1
        if expected_value < value_threshold: return False, -1, 0
        
        return True, best_option, STAKE_PER_BET

    def run_advanced_backtest(self, backtest_dir: Path, confidence_thresholds: list, value_thresholds: list):
        """
        Runs a backtest that places one bet per match at the first moment conditions are met.
        """
        json_files = list(backtest_dir.glob('*.json'))
        print(f"\nRunning EVENT-DRIVEN backtest on {len(json_files)} matches...")
        
        for conf_threshold in confidence_thresholds:
            for value_threshold in value_thresholds:
                print(f"\nTesting confidence>{conf_threshold:.2f}, value>{value_threshold:.2f}")
                strategy_results = []
                for file_path in tqdm(json_files, desc="Backtesting", leave=False):
                    with open(file_path, 'r') as f:
                        match_data = json.load(f)
                    
                    df_xgb, outcome = self.stax_model.process_match_for_xgb(match_data)
                    lstm_features_df = self.stax_model.get_lstm_features_df(match_data)
                    
                    if df_xgb is None or df_xgb.empty or lstm_features_df is None or lstm_features_df.empty or outcome == -1:
                        continue

                    # UPDATED: Iterate through each time step of the match up to the cutoff point
                    iteration_limit = min(len(df_xgb), MAX_ITERATION_INDEX)
                    for idx in range(SEQUENCE_LENGTH, iteration_limit):
                        try:
                            meta_features = self.get_meta_features_for_row(df_xgb, lstm_features_df, idx)
                            if meta_features is None: continue
                            
                            stax_probs = self.predict_with_best_model(meta_features)
                            
                            target_row = df_xgb.iloc[idx]
                            odds = [target_row['avg_home_odds'], target_row['avg_away_odds'], target_row['avg_draw_odds']]
                            
                            should_bet, bet_choice, stake = self.should_place_bet(stax_probs, odds, conf_threshold, value_threshold)
                            
                            if should_bet:
                                is_correct = (bet_choice == outcome)
                                pnl = (stake * odds[bet_choice] - stake) if is_correct else -stake
                                strategy_results.append({
                                    'bet_time_min': round(target_row['time_elapsed_s'] / 60),
                                    'conf_threshold': conf_threshold,
                                    'value_threshold': value_threshold,
                                    'prediction': bet_choice,
                                    'actual': outcome, 'correct': is_correct, 'pnl': pnl, 'stake': stake,
                                    'odds': odds[bet_choice], 'confidence': stax_probs[bet_choice],
                                    'expected_value': (stax_probs[bet_choice] * odds[bet_choice]) - 1
                                })
                                # Bet was placed, so we stop processing this match and move to the next.
                                break 
                        except Exception:
                            continue
                
                if strategy_results:
                    self.results.extend(strategy_results)
        
        self.analyze_comprehensive_results()

    def get_meta_features_for_row(self, df_xgb, lstm_features_df, target_idx):
        try:
            target_row = df_xgb.iloc[target_idx]
            
            lr_feats = target_row[LR_FEATURES]
            lr_scaled = self.stax_model.scalers['lr'].transform(lr_feats.values.reshape(1, -1))
            pred_lr = self.stax_model.models['lr'].predict_proba(lr_scaled)[0]
            
            xgb_feats = target_row[XGB_FEATURES]
            xgb_scaled = self.stax_model.scalers['xgb'].transform(xgb_feats.values.reshape(1, -1))
            pred_xgb = self.stax_model.models['xgb'].predict_proba(xgb_scaled)[0]
            
            start_idx = target_idx - SEQUENCE_LENGTH + 1
            sequence_df = lstm_features_df.iloc[start_idx:target_idx+1]
            if len(sequence_df) != SEQUENCE_LENGTH: return None
            
            sequence_np = sequence_df.values
            scaled_sequence = self.stax_model.scalers['lstm'].transform(sequence_np)
            X_lstm = scaled_sequence.reshape(1, SEQUENCE_LENGTH, sequence_np.shape[1])
            pred_lstm = self.stax_model.models['lstm'].predict(X_lstm, verbose=0)[0]
            
            entropy, std_dev, max_diff = self.stax_model.calculate_model_disagreement(pred_lr, pred_xgb, pred_lstm)
            
            meta_features = {
                'p_lr_H': pred_lr[0], 'p_lr_A': pred_lr[1], 'p_lr_D': pred_lr[2],
                'p_xgb_H': pred_xgb[0], 'p_xgb_A': pred_xgb[1], 'p_xgb_D': pred_xgb[2],
                'p_lstm_H': pred_lstm[0], 'p_lstm_A': pred_lstm[1], 'p_lstm_D': pred_lstm[2],
                'avg_H': (pred_lr[0] + pred_xgb[0] + pred_lstm[0]) / 3,
                'avg_A': (pred_lr[1] + pred_xgb[1] + pred_lstm[1]) / 3,
                'avg_D': (pred_lr[2] + pred_xgb[2] + pred_lstm[2]) / 3,
                'entropy': entropy, 'std_dev': std_dev, 'max_disagreement': max_diff,
                'time_factor': min(target_idx / 90, 1.0),
                'market_overround': (1/target_row['avg_home_odds'] + 1/target_row['avg_away_odds'] + 1/target_row['avg_draw_odds']) - 1,
                'score_diff': target_row['score_diff']
            }
            return pd.DataFrame([meta_features])
        except Exception:
            return None

    def predict_with_best_model(self, meta_features: pd.DataFrame) -> np.ndarray:
        method = self.stax_model.ensemble_method
        if method == 'lr':
            X_scaled = self.stax_model.meta_scalers['standard'].transform(meta_features)
            return self.stax_model.meta_models['lr'].predict_proba(X_scaled)[0]
        elif method == 'rf':
            return self.stax_model.meta_models['rf'].predict_proba(meta_features)[0]
        elif method == 'weighted_rf':
            X_weighted = self.stax_model.create_weighted_features(meta_features)
            return self.stax_model.meta_models['weighted_rf'].predict_proba(X_weighted)[0]
        else:
            weights = self.stax_model.model_weights or [0.33, 0.34, 0.33]
            pred = (weights[0] * meta_features[['p_lr_H', 'p_lr_A', 'p_lr_D']].values +
                    weights[1] * meta_features[['p_xgb_H', 'p_xgb_A', 'p_xgb_D']].values +
                    weights[2] * meta_features[['p_lstm_H', 'p_lstm_A', 'p_lstm_D']].values)
            return pred[0] / pred[0].sum()

    def analyze_comprehensive_results(self):
        if not self.results:
            print("No results to analyze.")
            return
        
        df_all = pd.DataFrame(self.results)
        threshold_summary = df_all.groupby(['conf_threshold', 'value_threshold']).agg(
            total_bets=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            total_staked=('stake', 'sum'),
            win_rate=('correct', 'mean')
        )
        threshold_summary['roi'] = (threshold_summary['total_pnl'] / threshold_summary['total_staked'] * 100).fillna(0)
        
        print("\n=== COMPREHENSIVE BACKTEST RESULTS (EVENT-DRIVEN) ===")
        print("\nPerformance by Threshold Combination:")
        print(threshold_summary.sort_values('roi', ascending=False).round(2).head(10))
        
        if threshold_summary.empty or threshold_summary.isna().all().all():
            print("\nNo profitable configurations found.")
            return

        best_config = threshold_summary['roi'].idxmax()
        print(f"\nBest Configuration: Confidence>{best_config[0]:.2f}, Value>{best_config[1]:.2f}")
        print(f"ROI: {threshold_summary.loc[best_config, 'roi']:.2f}%")
        
        best_df = df_all[(df_all['conf_threshold'] == best_config[0]) & (df_all['value_threshold'] == best_config[1])]
        
        print(f"\nAnalysis of Bet Timing for Best Configuration:")
        print(best_df['bet_time_min'].describe().round(2))

def main():
    """Main function to orchestrate the backtesting-only pipeline."""
    parser = argparse.ArgumentParser(description='Stax Meta-Model Event-Driven Backtester')
    parser.add_argument('--conf_thresholds', nargs='+', type=float, default=[0.55, 0.60, 0.65, 0.70, 0.75], help='Confidence thresholds to test')
    parser.add_argument('--value_thresholds', nargs='+', type=float, default=[0.00, 0.05, 0.10, 0.15, 0.20], help='Value thresholds to test')
    args = parser.parse_args()
    
    print("=== Starting Event-Driven Flat Stake Backtest Pipeline ===")
    
    stax = EnhancedStaxModel()
    stax.load_saved_models()
    
    backtester = ValueBettingBacktester(stax)
    backtester.run_advanced_backtest(
        DATA_DIR / "Backtest", 
        args.conf_thresholds,
        args.value_thresholds
    )

    print("\n--- Stax Event-Driven Backtest Complete ---")

if __name__ == '__main__':
    main()
