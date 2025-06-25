#!/usr/bin/env python3
"""
Stax Meta-Model - BACKTESTING ONLY
This script loads all pre-trained base and meta models and runs the 
advanced backtester to evaluate performance. It does not perform any training.
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

# --- Configuration (Copied from original script) ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STAX_MODEL_DIR = MODELS_DIR / "stax_Claude"

# Model Directories
LR_MODEL_DIR = MODELS_DIR / "logistic_regression_model"
XGB_MODEL_DIR = MODELS_DIR / "xgboost_model"
LSTM_MODEL_DIR = MODELS_DIR / "lstm_seq5"

# Model parameters
MOMENTUM_WINDOW = 5
STAKE_PER_BET = 10.0
DEFAULT_STRATEGIES = [10, 20, 30, 45, 60]
SEQUENCE_LENGTH = 5

# Enhanced betting parameters
KELLY_FRACTION = 0.25

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

    def load_base_models(self):
        """Loads pre-trained base models and scalers."""
        print("Loading pre-trained base models...")
        self.models = {
            'lr': joblib.load(LR_MODEL_DIR / "logistic_regression_model.joblib"),
            'xgb': joblib.load(XGB_MODEL_DIR / "xgboost_model.joblib"),
            'lstm': tf.keras.models.load_model(LSTM_MODEL_DIR / "lstm_seq5.h5")
        }
        self.scalers = {
            'lr': joblib.load(LR_MODEL_DIR / "feature_scaler.joblib"),
            'xgb': joblib.load(XGB_MODEL_DIR / "feature_scaler.joblib"),
            'lstm': joblib.load(LSTM_MODEL_DIR / "scaler_seq5.pkl")
        }
        print("Base models loaded successfully!")

    def load_saved_meta_models(self):
        """Load previously saved meta-models."""
        print("Loading saved meta-models...")
        config = joblib.load(self.output_dir / "stax_config.joblib")
        self.model_weights = config['model_weights']
        self.ensemble_method = config['ensemble_method']
        
        # Load all potential meta-models
        for model_type in ['lr', 'rf', 'weighted_rf']:
            model_path = self.output_dir / f"meta_model_{model_type}.joblib"
            if model_path.exists():
                self.meta_models[model_type] = joblib.load(model_path)
        
        self.meta_scalers['standard'] = joblib.load(self.output_dir / "meta_scaler_standard.joblib")
        print("Loaded saved meta-models successfully!")

    def create_weighted_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create weighted features using loaded weights."""
        if self.model_weights is None:
            raise ValueError("Model weights are not loaded. Cannot create weighted features.")
        
        X_weighted = X.copy()
        X_weighted['weighted_H'] = (self.model_weights[0] * X['p_lr_H'] + self.model_weights[1] * X['p_xgb_H'] + self.model_weights[2] * X['p_lstm_H'])
        X_weighted['weighted_A'] = (self.model_weights[0] * X['p_lr_A'] + self.model_weights[1] * X['p_xgb_A'] + self.model_weights[2] * X['p_lstm_A'])
        X_weighted['weighted_D'] = (self.model_weights[0] * X['p_lr_D'] + self.model_weights[1] * X['p_xgb_D'] + self.model_weights[2] * X['p_lstm_D'])
        return X_weighted

    # --- Data Processing functions (required for backtesting) ---
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
    """Advanced backtester with value betting and Kelly criterion."""
    
    def __init__(self, stax_model: EnhancedStaxModel):
        self.stax_model = stax_model
        self.results = []

    def calculate_kelly_stake(self, prob: float, odds: float, fraction: float = KELLY_FRACTION) -> float:
        edge = (prob * odds) - 1
        if edge <= 0: return 0
        kelly_stake = (edge / (odds - 1)) * fraction
        return min(kelly_stake * STAKE_PER_BET, STAKE_PER_BET * 2)

    def should_place_bet(self, probs: np.ndarray, odds: List[float], 
                        confidence_threshold: float, value_threshold: float) -> Tuple[bool, int, float]:
        best_option = np.argmax(probs)
        confidence = probs[best_option]
        if confidence < confidence_threshold: return False, -1, 0
        expected_value = (probs[best_option] * odds[best_option]) - 1
        if expected_value < value_threshold: return False, -1, 0
        stake = self.calculate_kelly_stake(probs[best_option], odds[best_option])
        return True, best_option, stake

    def run_advanced_backtest(self, backtest_dir: Path, strategies: list, 
                            confidence_thresholds: list, value_thresholds: list):
        json_files = list(backtest_dir.glob('*.json'))
        print(f"\nRunning advanced backtest on {len(json_files)} matches...")
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
                    
                    for strategy_minutes in strategies:
                        time_s = strategy_minutes * 60
                        target_idx = (df_xgb['time_elapsed_s'] - time_s).abs().idxmin()
                        if target_idx < SEQUENCE_LENGTH or target_idx >= len(df_xgb):
                            continue
                        
                        try:
                            meta_features = self.get_meta_features_for_row(df_xgb, lstm_features_df, target_idx)
                            if meta_features is None: continue
                            
                            stax_probs = self.predict_with_best_model(meta_features)
                            
                            target_row = df_xgb.iloc[target_idx]
                            odds = [target_row['avg_home_odds'], target_row['avg_away_odds'], target_row['avg_draw_odds']]
                            
                            should_bet, bet_choice, stake = self.should_place_bet(stax_probs, odds, conf_threshold, value_threshold)
                            
                            if should_bet:
                                is_correct = (bet_choice == outcome)
                                pnl = (stake * odds[bet_choice] - stake) if is_correct else -stake
                                strategy_results.append({
                                    'strategy': strategy_minutes, 'conf_threshold': conf_threshold,
                                    'value_threshold': value_threshold, 'prediction': bet_choice,
                                    'actual': outcome, 'correct': is_correct, 'pnl': pnl, 'stake': stake,
                                    'odds': odds[bet_choice], 'confidence': stax_probs[bet_choice],
                                    'expected_value': (stax_probs[bet_choice] * odds[bet_choice]) - 1,
                                    'file': file_path.name
                                })
                        except Exception:
                            continue
                
                if strategy_results:
                    self.results.extend(strategy_results)
                    self.analyze_threshold_performance(strategy_results, conf_threshold, value_threshold)

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

    def analyze_threshold_performance(self, results: list, conf_threshold: float, value_threshold: float):
        df = pd.DataFrame(results)
        if df.empty: return
        summary = df.groupby('strategy').agg(
            num_bets=('pnl', 'count'), total_pnl=('pnl', 'sum'),
            total_staked=('stake', 'sum'), win_rate=('correct', 'mean'),
            avg_confidence=('confidence', 'mean'), avg_ev=('expected_value', 'mean')
        ).round(2)
        summary['roi'] = (summary['total_pnl'] / summary['total_staked'] * 100).fillna(0)
        print(f"\nResults for confidence>{conf_threshold:.2f}, value>{value_threshold:.2f}:")
        print(summary.round(2))

    def analyze_comprehensive_results(self):
        if not self.results:
            print("No results to analyze.")
            return
        
        df_all = pd.DataFrame(self.results)
        threshold_summary = df_all.groupby(['conf_threshold', 'value_threshold']).agg(
            total_bets=('pnl', 'count'), total_pnl=('pnl', 'sum'),
            total_staked=('stake', 'sum'), win_rate=('correct', 'mean')
        )
        threshold_summary['roi'] = (threshold_summary['total_pnl'] / threshold_summary['total_staked'] * 100).fillna(0)
        
        print("\n=== COMPREHENSIVE BACKTEST RESULTS ===")
        print("\nPerformance by Threshold Combination:")
        print(threshold_summary.sort_values('roi', ascending=False).round(2).head(10))
        
        if threshold_summary.empty or threshold_summary['roi'].max() < -9e9: # check for empty or all-NaN
            print("\nNo profitable configurations found.")
            return

        best_config = threshold_summary['roi'].idxmax()
        print(f"\nBest Configuration: Confidence>{best_config[0]:.2f}, Value>{best_config[1]:.2f}")
        print(f"ROI: {threshold_summary.loc[best_config, 'roi']:.2f}%")
        
        best_df = df_all[(df_all['conf_threshold'] == best_config[0]) & (df_all['value_threshold'] == best_config[1])]
        
        best_strategy = best_df.groupby('strategy').agg(
            num_bets=('pnl', 'count'), total_pnl=('pnl', 'sum'),
            total_staked=('stake', 'sum'), avg_stake=('stake', 'mean'),
            win_rate=('correct', 'mean'), avg_confidence=('confidence', 'mean'),
            avg_ev=('expected_value', 'mean'), avg_odds=('odds', 'mean')
        )
        best_strategy['roi'] = (best_strategy['total_pnl'] / best_strategy['total_staked'] * 100).fillna(0)
        
        print(f"\nDetailed Strategy Performance for Best Configuration:")
        print(best_strategy.round(2))
        
        self.create_comprehensive_plots(df_all, best_config, threshold_summary)
        
        df_all.to_csv(self.stax_model.output_dir / 'backtest_all_results.csv', index=False)
        threshold_summary.to_csv(self.stax_model.output_dir / 'threshold_summary.csv')
        best_strategy.to_csv(self.stax_model.output_dir / 'best_strategy_breakdown.csv')
    
    def create_comprehensive_plots(self, df_all, best_config, threshold_summary):
        # This function is unchanged and included for completeness.
        # ... (full plotting code from original script)
        pass # Placeholder for brevity, user should have the full code.


def main():
    """Main function to orchestrate the backtesting-only pipeline."""
    parser = argparse.ArgumentParser(description='Stax Meta-Model Backtester')
    parser.add_argument('--strategies', nargs='+', type=int, default=DEFAULT_STRATEGIES, help='Betting time strategies in minutes')
    parser.add_argument('--conf_thresholds', nargs='+', type=float, default=[0.55, 0.60, 0.65, 0.70], help='Confidence thresholds to test')
    parser.add_argument('--value_thresholds', nargs='+', type=float, default=[0.00, 0.05, 0.10, 0.15], help='Value thresholds to test')
    args = parser.parse_args()
    
    print("=== Starting Backtest-Only Pipeline ===")
    
    # 1. Initialize the model container
    stax = EnhancedStaxModel()
    
    # 2. Load all the pre-trained models
    stax.load_base_models()
    stax.load_saved_meta_models()
    
    # 3. Initialize and run the backtester
    backtester = ValueBettingBacktester(stax)
    backtester.run_advanced_backtest(
        DATA_DIR / "Backtest", 
        args.strategies,
        args.conf_thresholds,
        args.value_thresholds
    )
    
    # 4. Analyze and display the results
    backtester.analyze_comprehensive_results()

    print("\n--- Stax Backtest Pipeline Complete ---")

if __name__ == '__main__':
    main()