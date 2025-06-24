#!/usr/bin/env python3
"""
Stax Meta-Model Pipeline
Combines predictions from Logistic Regression, XGBoost, and LSTM models
to improve sports match outcome predictions.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import xgboost as xgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STAX_MODEL_DIR = MODELS_DIR / "stax"  # Different from your script to avoid overwriting

# Model parameters
MOMENTUM_WINDOW = 5
STAKE_PER_BET = 10.0
DEFAULT_STRATEGIES = [10, 20, 30, 45, 60]
SEQUENCE_LENGTH = 5  # For LSTM

# Feature lists (from your script)
LR_FEATURES = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s', 'home_score', 'away_score', 'score_diff', 'avg_home_odds',
    'avg_away_odds', 'avg_draw_odds', 'std_home_odds', 'std_away_odds',
    'std_draw_odds', 'home_odds_momentum', 'away_odds_momentum',
    'draw_odds_momentum', 'prob_home', 'prob_away', 'prob_draw'
]


class StaxMetaModel:
    """Stax meta-model that combines predictions from multiple base models."""
    
    def __init__(self):
        self.output_dir = STAX_MODEL_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base models and scalers
        print("Loading base models...")
        self.models = {
            'lr': joblib.load(MODELS_DIR / "logistic_regression_model" / "logistic_regression_model.joblib"),
            'xgb': joblib.load(MODELS_DIR / "xgboost_model" / "xgboost_model.joblib"),
            'lstm': tf.keras.models.load_model(MODELS_DIR / "lstm_seq5" / "lstm_seq5.h5")
        }
        
        self.scalers = {
            'lr': joblib.load(MODELS_DIR / "logistic_regression_model" / "feature_scaler.joblib"),
            'xgb': joblib.load(MODELS_DIR / "xgboost_model" / "feature_scaler.joblib"),
            'lstm': joblib.load(MODELS_DIR / "lstm_seq5" / "scaler_seq5.pkl")
        }
        
        # Meta-model
        self.meta_model = None
        self.meta_scaler = StandardScaler()
        
        print("Base models loaded successfully!")
    
    def get_team_names(self, match_name: str, odds_data: dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract team names from match data."""
        if ' vs ' in match_name:
            teams = match_name.split(' vs ')
            return teams[0].strip(), teams[1].strip()
        
        non_draw_keys = [k for odds in odds_data.values() for k in odds if k.lower() != 'draw']
        if len(non_draw_keys) >= 2:
            return non_draw_keys[0], non_draw_keys[1]
        return None, None
    
    def process_match_for_lr(self, json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
        """Process match data for Logistic Regression model."""
        if not json_data:
            return None, -1
        
        match_name = json_data[0].get('match', 'Unknown')
        home_team, away_team = self.get_team_names(match_name, json_data[0].get('odds', {}))
        if not home_team or not away_team:
            return None, -1

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
                'home_score': h_score,
                'away_score': a_score,
                'avg_home_odds': np.mean([o for o in h_odds if o]),
                'avg_away_odds': np.mean([o for o in a_odds if o]),
                'avg_draw_odds': np.mean([o for o in d_odds if o]),
            })
        
        df = pd.DataFrame(rows).dropna()
        return df, final_outcome
    
    def process_match_for_xgb(self, json_data: List[Dict]) -> Tuple[Optional[pd.DataFrame], int]:
        """Process match data for XGBoost model."""
        df, final_outcome = self.process_match_for_lr(json_data)
        if df is None or df.empty:
            return None, -1
        
        # Add XGBoost-specific features
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
        """Get LSTM features dataframe."""
        df, _ = self.process_match_for_lr(json_data)
        if df is None or df.empty:
            return None

        lstm_feats_df = df[['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].copy()
        lstm_feats_df['score_diff'] = df['home_score'] - df['away_score']
        return lstm_feats_df
    
    def generate_meta_features(self, data_path: Path) -> pd.DataFrame:
        """Generate meta-features from base model predictions."""
        print(f"Generating meta-features from data in: {data_path}")
        json_files = list(data_path.glob('*.json'))[:180]  # Limit as in original LSTM script
        meta_data = []

        for file in tqdm(json_files, desc="Processing matches for meta-features"):
            with open(file, 'r') as f:
                match_data = json.load(f)[:180]  # Limit entries as in original

            df_lr, outcome = self.process_match_for_lr(match_data)
            df_xgb, _ = self.process_match_for_xgb(match_data)
            lstm_features_df = self.get_lstm_features_df(match_data)
            
            if df_lr is None or df_xgb is None or lstm_features_df is None:
                continue
            
            # Process each time point where we can generate all predictions
            for idx in range(SEQUENCE_LENGTH, min(len(df_lr), len(df_xgb), len(lstm_features_df))):
                try:
                    # LR prediction
                    lr_feats = df_lr.iloc[idx][LR_FEATURES]
                    lr_scaled = self.scalers['lr'].transform(lr_feats.values.reshape(1, -1))
                    pred_lr = self.models['lr'].predict_proba(lr_scaled)[0]
                    
                    # XGB prediction
                    xgb_feats = df_xgb.iloc[idx][XGB_FEATURES]
                    xgb_scaled = self.scalers['xgb'].transform(xgb_feats.values.reshape(1, -1))
                    pred_xgb = self.models['xgb'].predict_proba(xgb_scaled)[0]
                    
                    # LSTM prediction
                    start_idx = idx - SEQUENCE_LENGTH + 1
                    sequence_df = lstm_features_df.iloc[start_idx:idx+1]
                    if len(sequence_df) != SEQUENCE_LENGTH:
                        continue
                    
                    sequence_np = sequence_df.values
                    scaled_sequence = self.scalers['lstm'].transform(sequence_np)
                    X_lstm = scaled_sequence.reshape(1, SEQUENCE_LENGTH, sequence_np.shape[1])
                    pred_lstm = self.models['lstm'].predict(X_lstm, verbose=0)[0]
                    
                    # Combine predictions
                    row = {
                        'p_lr_H': pred_lr[0], 'p_lr_A': pred_lr[1], 'p_lr_D': pred_lr[2],
                        'p_xgb_H': pred_xgb[0], 'p_xgb_A': pred_xgb[1], 'p_xgb_D': pred_xgb[2],
                        'p_lstm_H': pred_lstm[0], 'p_lstm_A': pred_lstm[1], 'p_lstm_D': pred_lstm[2],
                        'final_outcome': outcome
                    }
                    meta_data.append(row)
                
                except Exception as e:
                    continue

        return pd.DataFrame(meta_data)
    
    def train_meta_model(self, X: pd.DataFrame, y: pd.Series):
        """Train the Stax meta-model."""
        print(f"Training meta-model on {len(X)} samples...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.meta_scaler.fit_transform(X_train)
        X_val_scaled = self.meta_scaler.transform(X_val)
        
        # Train meta-model with more sophisticated hyperparameters
        self.meta_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            C=1.0,  # Regularization
            random_state=42
        )
        
        self.meta_model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_pred = self.meta_model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        
        # Evaluate on validation set
        val_pred = self.meta_model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        
        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("\nValidation Classification Report:")
        print(classification_report(y_val, val_pred, 
                                  target_names=['Home Win', 'Away Win', 'Draw']))
        
        # Feature importance analysis
        self.analyze_feature_importance()
        
        # Save models
        self.save_models()
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance."""
        feature_names = [
            'LR_Home', 'LR_Away', 'LR_Draw',
            'XGB_Home', 'XGB_Away', 'XGB_Draw',
            'LSTM_Home', 'LSTM_Away', 'LSTM_Draw'
        ]
        
        # Get coefficients for each class
        coefs = self.meta_model.coef_
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        outcomes = ['Home Win', 'Away Win', 'Draw']
        
        for i, (coef, outcome, ax) in enumerate(zip(coefs, outcomes, axes)):
            sorted_idx = np.argsort(np.abs(coef))[::-1]
            ax.barh(range(len(coef)), coef[sorted_idx])
            ax.set_yticks(range(len(coef)))
            ax.set_yticklabels([feature_names[j] for j in sorted_idx])
            ax.set_xlabel('Coefficient Value')
            ax.set_title(f'Feature Importance for {outcome}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()
    
    def save_models(self):
        """Save the meta-model and scaler."""
        joblib.dump(self.meta_model, self.output_dir / "stax_meta_model.joblib")
        joblib.dump(self.meta_scaler, self.output_dir / "stax_meta_scaler.joblib")
        print(f"\nSaved meta-model to {self.output_dir}")
    
    def load_saved_models(self):
        """Load previously saved meta-model."""
        self.meta_model = joblib.load(self.output_dir / "stax_meta_model.joblib")
        self.meta_scaler = joblib.load(self.output_dir / "stax_meta_scaler.joblib")
        print("Loaded saved meta-model")


class StaxBacktester:
    """Backtester for the Stax meta-model."""
    
    def __init__(self, stax_model: StaxMetaModel):
        self.stax_model = stax_model
        self.results = []
    
    def run_backtest(self, backtest_dir: Path, strategies: list):
        """Run backtest for multiple time strategies."""
        json_files = list(backtest_dir.glob('*.json'))
        
        print(f"\nRunning backtest on {len(json_files)} matches...")
        
        for file_path in tqdm(json_files, desc="Backtesting"):
            with open(file_path, 'r') as f:
                match_data = json.load(f)[:180]
            
            df_xgb, outcome = self.stax_model.process_match_for_xgb(match_data)
            lstm_features_df = self.stax_model.get_lstm_features_df(match_data)
            
            if df_xgb is None or lstm_features_df is None or outcome == -1:
                continue
            
            for strategy_minutes in strategies:
                time_s = strategy_minutes * 60
                if time_s > (len(df_xgb) - 1) * 40:
                    continue
                
                # Find the row closest to target time
                target_idx = (df_xgb['time_elapsed_s'] - time_s).abs().idxmin()
                target_row = df_xgb.iloc[target_idx]
                
                # Skip if not enough data for LSTM
                if target_idx < SEQUENCE_LENGTH:
                    continue
                
                try:
                    # Generate base predictions
                    # LR
                    lr_feats = target_row[LR_FEATURES]
                    lr_scaled = self.stax_model.scalers['lr'].transform(lr_feats.values.reshape(1, -1))
                    pred_lr = self.stax_model.models['lr'].predict_proba(lr_scaled)[0]
                    
                    # XGB
                    xgb_feats = target_row[XGB_FEATURES]
                    xgb_scaled = self.stax_model.scalers['xgb'].transform(xgb_feats.values.reshape(1, -1))
                    pred_xgb = self.stax_model.models['xgb'].predict_proba(xgb_scaled)[0]
                    
                    # LSTM
                    start_idx = target_idx - SEQUENCE_LENGTH + 1
                    sequence_df = lstm_features_df.iloc[start_idx:target_idx+1]
                    if len(sequence_df) != SEQUENCE_LENGTH:
                        continue
                    
                    sequence_np = sequence_df.values
                    scaled_sequence = self.stax_model.scalers['lstm'].transform(sequence_np)
                    X_lstm = scaled_sequence.reshape(1, SEQUENCE_LENGTH, sequence_np.shape[1])
                    pred_lstm = self.stax_model.models['lstm'].predict(X_lstm, verbose=0)[0]
                    
                    # Generate Stax prediction
                    meta_features = np.concatenate([pred_lr, pred_xgb, pred_lstm]).reshape(1, -1)
                    meta_scaled = self.stax_model.meta_scaler.transform(meta_features)
                    stax_probs = self.stax_model.meta_model.predict_proba(meta_scaled)[0]
                    
                    bet_choice = np.argmax(stax_probs)
                    odds_map = {0: 'avg_home_odds', 1: 'avg_away_odds', 2: 'avg_draw_odds'}
                    bet_odds = target_row[odds_map[bet_choice]]
                    
                    # Calculate P&L
                    is_correct = (bet_choice == outcome)
                    pnl = (STAKE_PER_BET * bet_odds - STAKE_PER_BET) if is_correct else -STAKE_PER_BET
                    
                    self.results.append({
                        'strategy': strategy_minutes,
                        'prediction': bet_choice,
                        'actual': outcome,
                        'correct': is_correct,
                        'pnl': pnl,
                        'bet_odds': bet_odds,
                        'confidence': stax_probs[bet_choice],
                        'file': file_path.name
                    })
                
                except Exception as e:
                    continue
    
    def analyze_results(self):
        """Analyze and display backtest results."""
        df_results = pd.DataFrame(self.results)
        
        if df_results.empty:
            print("No backtest results to analyze.")
            return None
        
        # Group by strategy
        summary = df_results.groupby('strategy').agg({
            'pnl': ['count', 'sum'],
            'correct': 'mean',
            'confidence': 'mean'
        })
        
        summary.columns = ['total_bets', 'total_pnl', 'win_rate', 'avg_confidence']
        summary['win_rate'] *= 100
        summary['total_staked'] = summary['total_bets'] * STAKE_PER_BET
        summary['roi'] = (summary['total_pnl'] / summary['total_staked']) * 100
        
        print("\n=== Backtest Results ===")
        print(summary.round(2))
        
        # Visualize results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # P&L by strategy
        ax1.bar(summary.index, summary['total_pnl'], color=['g' if x > 0 else 'r' for x in summary['total_pnl']])
        ax1.set_xlabel('Strategy (minutes)')
        ax1.set_ylabel('Total P&L (£)')
        ax1.set_title('Profit/Loss by Strategy')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Win rate and ROI
        ax2_twin = ax2.twinx()
        
        ax2.plot(summary.index, summary['win_rate'], 'b-o', label='Win Rate', linewidth=2)
        ax2.set_ylabel('Win Rate (%)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax2_twin.plot(summary.index, summary['roi'], 'r-s', label='ROI', linewidth=2)
        ax2_twin.set_ylabel('ROI (%)', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        ax2.set_xlabel('Strategy (minutes)')
        ax2.set_title('Win Rate and ROI by Strategy')
        ax2.grid(True, alpha=0.3)
        
        # Confidence vs Accuracy
        ax3.scatter(summary['avg_confidence'] * 100, summary['win_rate'])
        ax3.set_xlabel('Average Confidence (%)')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Model Confidence vs Actual Win Rate')
        ax3.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect Calibration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Distribution of predictions
        pred_dist = df_results.groupby(['strategy', 'prediction']).size().unstack(fill_value=0)
        pred_dist.plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_xlabel('Strategy (minutes)')
        ax4.set_ylabel('Number of Bets')
        ax4.set_title('Distribution of Predictions')
        ax4.legend(title='Prediction', labels=['Home Win', 'Away Win', 'Draw'])
        
        plt.tight_layout()
        plt.savefig(self.stax_model.output_dir / 'backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        summary.to_csv(self.stax_model.output_dir / 'backtest_summary.csv')
        df_results.to_csv(self.stax_model.output_dir / 'backtest_detailed.csv', index=False)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Stax Meta-Model Pipeline')
    parser.add_argument('--mode', choices=['train', 'backtest', 'all'], 
                       default='all', help='Operation mode')
    parser.add_argument('--strategies', nargs='+', type=int, 
                       default=DEFAULT_STRATEGIES,
                       help='Betting time strategies in minutes')
    
    args = parser.parse_args()
    
    # Initialize Stax model
    stax = StaxMetaModel()
    
    if args.mode in ['train', 'all']:
        print("=== Training Stax Meta-Model ===")
        
        # Generate meta-features from training data
        meta_df = stax.generate_meta_features(DATA_DIR / "Training")
        
        if meta_df.empty:
            print("No meta-features generated. Exiting.")
            return
        
        print(f"Generated {len(meta_df)} samples for training")
        
        # Prepare data
        X_meta = meta_df.drop('final_outcome', axis=1)
        y_meta = meta_df['final_outcome']
        
        # Train meta-model
        stax.train_meta_model(X_meta, y_meta)
    
    if args.mode in ['backtest', 'all']:
        # Load trained model if only backtesting
        if args.mode == 'backtest':
            stax.load_saved_models()
        
        print("\n=== Running Backtest ===")
        backtester = StaxBacktester(stax)
        backtester.run_backtest(DATA_DIR / "Backtest", args.strategies)
        summary = backtester.analyze_results()
        
        if summary is not None:
            print(f"\nResults saved to {stax.output_dir}")
    
    print("\n--- Stax Meta-Model Pipeline Complete ---")


if __name__ == '__main__':
    main()