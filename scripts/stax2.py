#!/usr/bin/env python3
"""
Stax Meta-Model Pipeline V2
Enhanced with value betting, confidence thresholds, and advanced ensemble strategies
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import xgboost as xgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STAX_MODEL_DIR = MODELS_DIR / "stax_Claude"

# Model parameters
MOMENTUM_WINDOW = 5
STAKE_PER_BET = 10.0
DEFAULT_STRATEGIES = [10, 20, 30, 45, 60]
SEQUENCE_LENGTH = 5  # For LSTM

# Enhanced betting parameters
MIN_CONFIDENCE_THRESHOLD = 0.65  # Only bet when confidence > 65%
MIN_VALUE_THRESHOLD = 0.05  # Only bet when expected value > 5%
KELLY_FRACTION = 0.25  # Use fractional Kelly criterion for bet sizing

# Feature lists
LR_FEATURES = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
XGB_FEATURES = [
    'time_elapsed_s', 'home_score', 'away_score', 'score_diff', 'avg_home_odds',
    'avg_away_odds', 'avg_draw_odds', 'std_home_odds', 'std_away_odds',
    'std_draw_odds', 'home_odds_momentum', 'away_odds_momentum',
    'draw_odds_momentum', 'prob_home', 'prob_away', 'prob_draw'
]


class EnhancedStaxModel:
    """Enhanced Stax meta-model with advanced ensemble techniques."""
    
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
        
        # Meta-models (we'll try multiple approaches)
        self.meta_models = {}
        self.meta_scalers = {}
        self.model_weights = None
        self.ensemble_method = 'weighted_rf'  # Can be 'lr', 'rf', 'weighted', 'weighted_rf'
        
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
    
    def calculate_model_disagreement(self, pred_lr, pred_xgb, pred_lstm):
        """Calculate disagreement metrics between models."""
        predictions = np.array([pred_lr, pred_xgb, pred_lstm])
        
        # Entropy of average prediction (uncertainty)
        avg_pred = predictions.mean(axis=0)
        entropy = -np.sum(avg_pred * np.log(avg_pred + 1e-10))
        
        # Standard deviation across models (disagreement)
        std_dev = predictions.std(axis=0).mean()
        
        # Max disagreement between any two models
        max_diff = 0
        for i in range(3):
            for j in range(i+1, 3):
                max_diff = max(max_diff, np.abs(predictions[i] - predictions[j]).max())
        
        return entropy, std_dev, max_diff
    
    def generate_enhanced_meta_features(self, data_path: Path) -> pd.DataFrame:
        """Generate enhanced meta-features with additional metrics."""
        print(f"Generating enhanced meta-features from data in: {data_path}")
        json_files = list(data_path.glob('*.json'))[:180]
        meta_data = []

        for file in tqdm(json_files, desc="Processing matches for meta-features"):
            with open(file, 'r') as f:
                match_data = json.load(f)[:180]

            df_lr, outcome = self.process_match_for_lr(match_data)
            df_xgb, _ = self.process_match_for_xgb(match_data)
            lstm_features_df = self.get_lstm_features_df(match_data)
            
            if df_lr is None or df_xgb is None or lstm_features_df is None:
                continue
            
            for idx in range(SEQUENCE_LENGTH, min(len(df_lr), len(df_xgb), len(lstm_features_df))):
                try:
                    # Get base model predictions
                    lr_feats = df_lr.iloc[idx][LR_FEATURES]
                    lr_scaled = self.scalers['lr'].transform(lr_feats.values.reshape(1, -1))
                    pred_lr = self.models['lr'].predict_proba(lr_scaled)[0]
                    
                    xgb_feats = df_xgb.iloc[idx][XGB_FEATURES]
                    xgb_scaled = self.scalers['xgb'].transform(xgb_feats.values.reshape(1, -1))
                    pred_xgb = self.models['xgb'].predict_proba(xgb_scaled)[0]
                    
                    start_idx = idx - SEQUENCE_LENGTH + 1
                    sequence_df = lstm_features_df.iloc[start_idx:idx+1]
                    if len(sequence_df) != SEQUENCE_LENGTH:
                        continue
                    
                    sequence_np = sequence_df.values
                    scaled_sequence = self.scalers['lstm'].transform(sequence_np)
                    X_lstm = scaled_sequence.reshape(1, SEQUENCE_LENGTH, sequence_np.shape[1])
                    pred_lstm = self.models['lstm'].predict(X_lstm, verbose=0)[0]
                    
                    # Calculate additional meta-features
                    entropy, std_dev, max_diff = self.calculate_model_disagreement(pred_lr, pred_xgb, pred_lstm)
                    
                    # Time decay feature (matches are more predictable later)
                    time_factor = min(idx / 90, 1.0)  # Normalize to [0, 1]
                    
                    # Odds-based features
                    home_odds = df_xgb.iloc[idx]['avg_home_odds']
                    away_odds = df_xgb.iloc[idx]['avg_away_odds']
                    draw_odds = df_xgb.iloc[idx]['avg_draw_odds']
                    
                    # Market confidence (lower total probability = more confident market)
                    market_overround = (1/home_odds + 1/away_odds + 1/draw_odds) - 1
                    
                    # Create enhanced feature row
                    row = {
                        # Original predictions
                        'p_lr_H': pred_lr[0], 'p_lr_A': pred_lr[1], 'p_lr_D': pred_lr[2],
                        'p_xgb_H': pred_xgb[0], 'p_xgb_A': pred_xgb[1], 'p_xgb_D': pred_xgb[2],
                        'p_lstm_H': pred_lstm[0], 'p_lstm_A': pred_lstm[1], 'p_lstm_D': pred_lstm[2],
                        
                        # Aggregated predictions
                        'avg_H': (pred_lr[0] + pred_xgb[0] + pred_lstm[0]) / 3,
                        'avg_A': (pred_lr[1] + pred_xgb[1] + pred_lstm[1]) / 3,
                        'avg_D': (pred_lr[2] + pred_xgb[2] + pred_lstm[2]) / 3,
                        
                        # Disagreement metrics
                        'entropy': entropy,
                        'std_dev': std_dev,
                        'max_disagreement': max_diff,
                        
                        # Additional features
                        'time_factor': time_factor,
                        'market_overround': market_overround,
                        'score_diff': df_xgb.iloc[idx]['score_diff'],
                        
                        # Target
                        'final_outcome': outcome
                    }
                    meta_data.append(row)
                
                except Exception as e:
                    continue

        return pd.DataFrame(meta_data)
    
    def calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate sample weights based on model agreement and confidence."""
        weights = np.ones(len(df))
        
        # Upweight samples where models agree
        avg_probs = df[['avg_H', 'avg_A', 'avg_D']].values
        max_probs = avg_probs.max(axis=1)
        weights *= (1 + max_probs)  # Higher weight for confident predictions
        
        # Downweight samples with high disagreement
        weights *= (1 - df['std_dev'] * 0.5)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights
    
    def train_meta_models(self, X: pd.DataFrame, y: pd.Series):
        """Train multiple meta-models and select the best."""
        print(f"Training enhanced meta-models on {len(X)} samples...")
        
        # Calculate sample weights
        sample_weights = self.calculate_sample_weights(X)
        
        # Split data
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Drop the target column if it exists
        feature_cols = [col for col in X.columns if col != 'final_outcome']
        X_train = X_train[feature_cols]
        X_val = X_val[feature_cols]
        
        # Initialize scalers
        self.meta_scalers['standard'] = StandardScaler()
        X_train_scaled = self.meta_scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.meta_scalers['standard'].transform(X_val)
        
        # 1. Logistic Regression
        print("\n1. Training Logistic Regression meta-model...")
        self.meta_models['lr'] = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            C=0.5,  # More regularization
            random_state=42
        )
        self.meta_models['lr'].fit(X_train_scaled, y_train, sample_weight=w_train)
        lr_pred = self.meta_models['lr'].predict(X_val_scaled)
        lr_acc = accuracy_score(y_val, lr_pred)
        print(f"LR Validation Accuracy: {lr_acc:.4f}")
        
        # 2. Random Forest
        print("\n2. Training Random Forest meta-model...")
        self.meta_models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.meta_models['rf'].fit(X_train, y_train, sample_weight=w_train)
        rf_pred = self.meta_models['rf'].predict(X_val)
        rf_acc = accuracy_score(y_val, rf_pred)
        print(f"RF Validation Accuracy: {rf_acc:.4f}")
        
        # 3. Weighted Average Ensemble
        print("\n3. Optimizing model weights...")
        self.optimize_model_weights(X_train, y_train, X_val, y_val)
        
        # 4. Weighted Random Forest (best of both worlds)
        print("\n4. Training Weighted RF with optimized features...")
        # Create weighted features
        X_train_weighted = self.create_weighted_features(X_train)
        X_val_weighted = self.create_weighted_features(X_val)
        
        self.meta_models['weighted_rf'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        )
        self.meta_models['weighted_rf'].fit(X_train_weighted, y_train, sample_weight=w_train)
        wrf_pred = self.meta_models['weighted_rf'].predict(X_val_weighted)
        wrf_acc = accuracy_score(y_val, wrf_pred)
        print(f"Weighted RF Validation Accuracy: {wrf_acc:.4f}")
        
        # Select best model
        accuracies = {'lr': lr_acc, 'rf': rf_acc, 'weighted_rf': wrf_acc}
        best_model = max(accuracies, key=accuracies.get)
        self.ensemble_method = best_model
        print(f"\nBest model: {best_model} with accuracy {accuracies[best_model]:.4f}")
        
        # Print detailed report for best model
        if best_model == 'lr':
            best_pred = lr_pred
        elif best_model == 'rf':
            best_pred = rf_pred
        else:
            best_pred = wrf_pred
            
        print("\nBest Model Classification Report:")
        print(classification_report(y_val, best_pred, 
                                  target_names=['Home Win', 'Away Win', 'Draw']))
        
        # Save all models
        self.save_models()
    
    def optimize_model_weights(self, X_train, y_train, X_val, y_val):
        """Optimize weights for model ensemble."""
        # Get individual model predictions
        lr_cols = ['p_lr_H', 'p_lr_A', 'p_lr_D']
        xgb_cols = ['p_xgb_H', 'p_xgb_A', 'p_xgb_D']
        lstm_cols = ['p_lstm_H', 'p_lstm_A', 'p_lstm_D']
        
        # Find optimal weights using validation set
        best_weights = None
        best_acc = 0
        
        for w1 in np.arange(0.1, 0.8, 0.1):
            for w2 in np.arange(0.1, 0.8, 0.1):
                w3 = 1 - w1 - w2
                if w3 <= 0 or w3 >= 1:
                    continue
                
                # Weighted predictions
                weighted_pred = (
                    w1 * X_val[lr_cols].values +
                    w2 * X_val[xgb_cols].values +
                    w3 * X_val[lstm_cols].values
                )
                
                pred_classes = np.argmax(weighted_pred, axis=1)
                acc = accuracy_score(y_val, pred_classes)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = [w1, w2, w3]
        
        self.model_weights = best_weights
        print(f"Optimal weights - LR: {best_weights[0]:.2f}, XGB: {best_weights[1]:.2f}, LSTM: {best_weights[2]:.2f}")
        print(f"Weighted ensemble accuracy: {best_acc:.4f}")
    
    def create_weighted_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create weighted features using optimized weights."""
        if self.model_weights is None:
            self.model_weights = [0.33, 0.34, 0.33]
        
        X_weighted = X.copy()
        
        # Add weighted predictions
        X_weighted['weighted_H'] = (
            self.model_weights[0] * X['p_lr_H'] +
            self.model_weights[1] * X['p_xgb_H'] +
            self.model_weights[2] * X['p_lstm_H']
        )
        X_weighted['weighted_A'] = (
            self.model_weights[0] * X['p_lr_A'] +
            self.model_weights[1] * X['p_xgb_A'] +
            self.model_weights[2] * X['p_lstm_A']
        )
        X_weighted['weighted_D'] = (
            self.model_weights[0] * X['p_lr_D'] +
            self.model_weights[1] * X['p_xgb_D'] +
            self.model_weights[2] * X['p_lstm_D']
        )
        
        return X_weighted
    
    def save_models(self):
        """Save all models and metadata."""
        # Save meta-models
        for name, model in self.meta_models.items():
            joblib.dump(model, self.output_dir / f"meta_model_{name}.joblib")
        
        # Save scalers
        for name, scaler in self.meta_scalers.items():
            joblib.dump(scaler, self.output_dir / f"meta_scaler_{name}.joblib")
        
        # Save weights and config
        config = {
            'model_weights': self.model_weights,
            'ensemble_method': self.ensemble_method,
            'min_confidence': MIN_CONFIDENCE_THRESHOLD,
            'min_value': MIN_VALUE_THRESHOLD,
            'kelly_fraction': KELLY_FRACTION
        }
        joblib.dump(config, self.output_dir / "stax_config.joblib")
        
        print(f"\nSaved all models to {self.output_dir}")
    
    def load_saved_models(self):
        """Load previously saved models."""
        # Load config
        config = joblib.load(self.output_dir / "stax_config.joblib")
        self.model_weights = config['model_weights']
        self.ensemble_method = config['ensemble_method']
        
        # Load meta-models
        for model_type in ['lr', 'rf', 'weighted_rf']:
            model_path = self.output_dir / f"meta_model_{model_type}.joblib"
            if model_path.exists():
                self.meta_models[model_type] = joblib.load(model_path)
        
        # Load scalers
        self.meta_scalers['standard'] = joblib.load(self.output_dir / "meta_scaler_standard.joblib")
        
        print("Loaded saved models")


class ValueBettingBacktester:
    """Advanced backtester with value betting and Kelly criterion."""
    
    def __init__(self, stax_model: EnhancedStaxModel):
        self.stax_model = stax_model
        self.results = []
        self.detailed_results = []
    
    def calculate_kelly_stake(self, prob: float, odds: float, fraction: float = KELLY_FRACTION) -> float:
        """Calculate optimal stake using Kelly criterion."""
        edge = (prob * odds) - 1
        if edge <= 0:
            return 0
        
        kelly_stake = (edge / (odds - 1)) * fraction
        # Cap at maximum stake
        return min(kelly_stake * STAKE_PER_BET, STAKE_PER_BET * 2)
    
    def should_place_bet(self, probs: np.ndarray, odds: List[float], 
                        confidence_threshold: float, value_threshold: float) -> Tuple[bool, int, float]:
        """Determine if we should place a bet based on value and confidence."""
        best_option = np.argmax(probs)
        confidence = probs[best_option]
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return False, -1, 0
        
        # Calculate expected value
        expected_value = (probs[best_option] * odds[best_option]) - 1
        
        # Check value threshold
        if expected_value < value_threshold:
            return False, -1, 0
        
        # Calculate stake
        stake = self.calculate_kelly_stake(probs[best_option], odds[best_option])
        
        return True, best_option, stake
    
    def run_advanced_backtest(self, backtest_dir: Path, strategies: list, 
                            confidence_thresholds: list = None,
                            value_thresholds: list = None):
        """Run backtest with multiple threshold configurations."""
        if confidence_thresholds is None:
            confidence_thresholds = [0.55, 0.60, 0.65, 0.70]
        if value_thresholds is None:
            value_thresholds = [0.00, 0.05, 0.10, 0.15]
        
        json_files = list(backtest_dir.glob('*.json'))
        print(f"\nRunning advanced backtest on {len(json_files)} matches...")
        
        # Test different threshold combinations
        for conf_threshold in confidence_thresholds:
            for value_threshold in value_thresholds:
                print(f"\nTesting confidence>{conf_threshold:.2f}, value>{value_threshold:.2f}")
                
                strategy_results = []
                
                for file_path in tqdm(json_files, desc="Backtesting", leave=False):
                    with open(file_path, 'r') as f:
                        match_data = json.load(f)[:180]  # Limit entries per match
                    
                    df_xgb, outcome = self.stax_model.process_match_for_xgb(match_data)
                    lstm_features_df = self.stax_model.get_lstm_features_df(match_data)
                    
                    if df_xgb is None or lstm_features_df is None or outcome == -1:
                        continue
                    
                    for strategy_minutes in strategies:
                        time_s = strategy_minutes * 60
                        if time_s > (len(df_xgb) - 1) * 40:
                            continue
                        
                        target_idx = (df_xgb['time_elapsed_s'] - time_s).abs().idxmin()
                        if target_idx < SEQUENCE_LENGTH:
                            continue
                        
                        try:
                            # Generate predictions (same as before)
                            target_row = df_xgb.iloc[target_idx]
                            
                            # Get base predictions
                            meta_features = self.get_meta_features_for_row(
                                df_xgb, lstm_features_df, target_idx, outcome
                            )
                            
                            if meta_features is None:
                                continue
                            
                            # Get Stax prediction
                            stax_probs = self.predict_with_best_model(meta_features)
                            
                            # Get odds
                            odds = [
                                target_row['avg_home_odds'],
                                target_row['avg_away_odds'],
                                target_row['avg_draw_odds']
                            ]
                            
                            # Check if we should bet
                            should_bet, bet_choice, stake = self.should_place_bet(
                                stax_probs, odds, conf_threshold, value_threshold
                            )
                            
                            if should_bet:
                                # Calculate P&L
                                is_correct = (bet_choice == outcome)
                                pnl = (stake * odds[bet_choice] - stake) if is_correct else -stake
                                
                                strategy_results.append({
                                    'strategy': strategy_minutes,
                                    'conf_threshold': conf_threshold,
                                    'value_threshold': value_threshold,
                                    'prediction': bet_choice,
                                    'actual': outcome,
                                    'correct': is_correct,
                                    'pnl': pnl,
                                    'stake': stake,
                                    'odds': odds[bet_choice],
                                    'confidence': stax_probs[bet_choice],
                                    'expected_value': (stax_probs[bet_choice] * odds[bet_choice]) - 1,
                                    'file': file_path.name
                                })
                        
                        except Exception as e:
                            continue
                
                # Store results for this threshold combination
                if strategy_results:
                    self.results.extend(strategy_results)
                    self.analyze_threshold_performance(strategy_results, conf_threshold, value_threshold)
    
    def get_meta_features_for_row(self, df_xgb, lstm_features_df, target_idx, outcome):
        """Extract meta features for a specific time point."""
        try:
            target_row = df_xgb.iloc[target_idx]
            
            # LR prediction
            lr_feats = target_row[LR_FEATURES]
            lr_scaled = self.stax_model.scalers['lr'].transform(lr_feats.values.reshape(1, -1))
            pred_lr = self.stax_model.models['lr'].predict_proba(lr_scaled)[0]
            
            # XGB prediction
            xgb_feats = target_row[XGB_FEATURES]
            xgb_scaled = self.stax_model.scalers['xgb'].transform(xgb_feats.values.reshape(1, -1))
            pred_xgb = self.stax_model.models['xgb'].predict_proba(xgb_scaled)[0]
            
            # LSTM prediction
            start_idx = target_idx - SEQUENCE_LENGTH + 1
            sequence_df = lstm_features_df.iloc[start_idx:target_idx+1]
            if len(sequence_df) != SEQUENCE_LENGTH:
                return None
            
            sequence_np = sequence_df.values
            scaled_sequence = self.stax_model.scalers['lstm'].transform(sequence_np)
            X_lstm = scaled_sequence.reshape(1, SEQUENCE_LENGTH, sequence_np.shape[1])
            pred_lstm = self.stax_model.models['lstm'].predict(X_lstm, verbose=0)[0]
            
            # Calculate meta-features
            entropy, std_dev, max_diff = self.stax_model.calculate_model_disagreement(
                pred_lr, pred_xgb, pred_lstm
            )
            
            # Create feature dictionary
            meta_features = {
                'p_lr_H': pred_lr[0], 'p_lr_A': pred_lr[1], 'p_lr_D': pred_lr[2],
                'p_xgb_H': pred_xgb[0], 'p_xgb_A': pred_xgb[1], 'p_xgb_D': pred_xgb[2],
                'p_lstm_H': pred_lstm[0], 'p_lstm_A': pred_lstm[1], 'p_lstm_D': pred_lstm[2],
                'avg_H': (pred_lr[0] + pred_xgb[0] + pred_lstm[0]) / 3,
                'avg_A': (pred_lr[1] + pred_xgb[1] + pred_lstm[1]) / 3,
                'avg_D': (pred_lr[2] + pred_xgb[2] + pred_lstm[2]) / 3,
                'entropy': entropy,
                'std_dev': std_dev,
                'max_disagreement': max_diff,
                'time_factor': min(target_idx / 90, 1.0),
                'market_overround': (1/target_row['avg_home_odds'] + 
                                   1/target_row['avg_away_odds'] + 
                                   1/target_row['avg_draw_odds']) - 1,
                'score_diff': target_row['score_diff']
            }
            
            return pd.DataFrame([meta_features])
            
        except Exception as e:
            return None
    
    def predict_with_best_model(self, meta_features: pd.DataFrame) -> np.ndarray:
        """Make prediction using the best ensemble method."""
        if self.stax_model.ensemble_method == 'lr':
            X_scaled = self.stax_model.meta_scalers['standard'].transform(meta_features)
            return self.stax_model.meta_models['lr'].predict_proba(X_scaled)[0]
        
        elif self.stax_model.ensemble_method == 'rf':
            return self.stax_model.meta_models['rf'].predict_proba(meta_features)[0]
        
        elif self.stax_model.ensemble_method == 'weighted_rf':
            X_weighted = self.stax_model.create_weighted_features(meta_features)
            return self.stax_model.meta_models['weighted_rf'].predict_proba(X_weighted)[0]
        
        else:  # Simple weighted average
            if self.stax_model.model_weights is None:
                self.stax_model.model_weights = [0.33, 0.34, 0.33]
            
            weighted_pred = (
                self.stax_model.model_weights[0] * meta_features[['p_lr_H', 'p_lr_A', 'p_lr_D']].values +
                self.stax_model.model_weights[1] * meta_features[['p_xgb_H', 'p_xgb_A', 'p_xgb_D']].values +
                self.stax_model.model_weights[2] * meta_features[['p_lstm_H', 'p_lstm_A', 'p_lstm_D']].values
            )
            return weighted_pred[0] / weighted_pred[0].sum()
    
    def analyze_threshold_performance(self, results: list, conf_threshold: float, value_threshold: float):
        """Analyze performance for specific threshold combination."""
        df = pd.DataFrame(results)
        if df.empty:
            return
        
        summary = df.groupby('strategy').agg({
            'pnl': ['count', 'sum', 'mean'],
            'stake': 'sum',
            'correct': 'mean',
            'confidence': 'mean',
            'expected_value': 'mean'
        })
        
        summary.columns = ['num_bets', 'total_pnl', 'avg_pnl', 'total_staked', 
                          'win_rate', 'avg_confidence', 'avg_ev']
        summary['roi'] = (summary['total_pnl'] / summary['total_staked'] * 100).fillna(0)
        
        print(f"\nResults for confidence>{conf_threshold:.2f}, value>{value_threshold:.2f}:")
        print(summary.round(2))
    
    def analyze_comprehensive_results(self):
        """Comprehensive analysis of all backtest results."""
        if not self.results:
            print("No results to analyze.")
            return
        
        df_all = pd.DataFrame(self.results)
        
        # Find best threshold combination
        threshold_summary = df_all.groupby(['conf_threshold', 'value_threshold']).agg({
            'pnl': ['count', 'sum'],
            'stake': 'sum',
            'correct': 'mean'
        })
        
        threshold_summary.columns = ['total_bets', 'total_pnl', 'total_staked', 'win_rate']
        threshold_summary['roi'] = (threshold_summary['total_pnl'] / 
                                   threshold_summary['total_staked'] * 100).fillna(0)
        threshold_summary = threshold_summary.round(2)
        
        print("\n=== COMPREHENSIVE BACKTEST RESULTS ===")
        print("\nPerformance by Threshold Combination:")
        print(threshold_summary.sort_values('roi', ascending=False).head(10))
        
        # Best performing configuration
        best_config = threshold_summary['roi'].idxmax()
        print(f"\nBest Configuration: Confidence>{best_config[0]:.2f}, Value>{best_config[1]:.2f}")
        print(f"ROI: {threshold_summary.loc[best_config, 'roi']:.2f}%")
        
        # Analyze best configuration in detail
        best_df = df_all[(df_all['conf_threshold'] == best_config[0]) & 
                        (df_all['value_threshold'] == best_config[1])]
        
        # Strategy breakdown for best config
        best_strategy = best_df.groupby('strategy').agg({
            'pnl': ['count', 'sum', 'mean'],
            'stake': ['sum', 'mean'],
            'correct': 'mean',
            'confidence': 'mean',
            'expected_value': 'mean',
            'odds': 'mean'
        })
        
        best_strategy.columns = ['num_bets', 'total_pnl', 'avg_pnl', 'total_staked', 
                                'avg_stake', 'win_rate', 'avg_confidence', 'avg_ev', 'avg_odds']
        best_strategy['roi'] = (best_strategy['total_pnl'] / best_strategy['total_staked'] * 100).fillna(0)
        
        print(f"\nDetailed Strategy Performance for Best Configuration:")
        print(best_strategy.round(2))
        
        # Visualizations
        self.create_comprehensive_plots(df_all, best_config, threshold_summary)
        
        # Save results
        df_all.to_csv(self.stax_model.output_dir / 'backtest_all_results.csv', index=False)
        threshold_summary.to_csv(self.stax_model.output_dir / 'threshold_summary.csv')
        best_strategy.to_csv(self.stax_model.output_dir / 'best_strategy_breakdown.csv')
        
        return best_config, threshold_summary
    
    def create_comprehensive_plots(self, df_all: pd.DataFrame, best_config: tuple, 
                                  threshold_summary: pd.DataFrame):
        """Create comprehensive visualization plots."""
        # Set up the plot
        fig = plt.figure(figsize=(20, 15))
        
        # 1. ROI Heatmap by Thresholds
        ax1 = plt.subplot(3, 3, 1)
        roi_pivot = threshold_summary['roi'].reset_index().pivot(
            index='conf_threshold', columns='value_threshold', values='roi'
        )
        sns.heatmap(roi_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax1)
        ax1.set_title('ROI (%) by Threshold Combination')
        ax1.set_xlabel('Value Threshold')
        ax1.set_ylabel('Confidence Threshold')
        
        # 2. Number of Bets Heatmap
        ax2 = plt.subplot(3, 3, 2)
        bets_pivot = threshold_summary['total_bets'].reset_index().pivot(
            index='conf_threshold', columns='value_threshold', values='total_bets'
        )
        sns.heatmap(bets_pivot, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Number of Bets by Threshold Combination')
        ax2.set_xlabel('Value Threshold')
        ax2.set_ylabel('Confidence Threshold')
        
        # 3. Win Rate Heatmap
        ax3 = plt.subplot(3, 3, 3)
        wr_pivot = (threshold_summary['win_rate'] * 100).reset_index().pivot(
            index='conf_threshold', columns='value_threshold', values='win_rate'
        )
        sns.heatmap(wr_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Win Rate (%) by Threshold Combination')
        ax3.set_xlabel('Value Threshold')
        ax3.set_ylabel('Confidence Threshold')
        
        # 4. P&L by Strategy (Best Config)
        ax4 = plt.subplot(3, 3, 4)
        best_df = df_all[(df_all['conf_threshold'] == best_config[0]) & 
                        (df_all['value_threshold'] == best_config[1])]
        strategy_pnl = best_df.groupby('strategy')['pnl'].sum()
        strategy_pnl.plot(kind='bar', ax=ax4, color=['g' if x > 0 else 'r' for x in strategy_pnl])
        ax4.set_title(f'P&L by Strategy (Best Config: C>{best_config[0]:.2f}, V>{best_config[1]:.2f})')
        ax4.set_xlabel('Strategy (minutes)')
        ax4.set_ylabel('Total P&L (£)')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 5. Expected Value vs Actual Returns
        ax5 = plt.subplot(3, 3, 5)
        ev_bins = np.linspace(best_df['expected_value'].min(), best_df['expected_value'].max(), 20)
        best_df['ev_bin'] = pd.cut(best_df['expected_value'], bins=ev_bins)
        ev_analysis = best_df.groupby('ev_bin').agg({
            'pnl': 'mean',
            'correct': 'mean',
            'stake': 'count'
        })
        
        ax5_twin = ax5.twinx()
        ax5.bar(range(len(ev_analysis)), ev_analysis['pnl'], alpha=0.7, label='Avg P&L')
        ax5_twin.plot(range(len(ev_analysis)), ev_analysis['correct'] * 100, 'r-o', label='Win Rate %')
        ax5.set_xlabel('Expected Value Bins')
        ax5.set_ylabel('Average P&L (£)')
        ax5_twin.set_ylabel('Win Rate (%)')
        ax5.set_title('EV vs Actual Returns')
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')
        
        # 6. Confidence Distribution
        ax6 = plt.subplot(3, 3, 6)
        best_df['confidence'].hist(bins=30, ax=ax6, alpha=0.7, color='blue')
        ax6.axvline(best_df['confidence'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {best_df['confidence'].mean():.3f}")
        ax6.set_xlabel('Model Confidence')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Distribution of Betting Confidence')
        ax6.legend()
        
        # 7. Cumulative P&L Over Time
        ax7 = plt.subplot(3, 3, 7)
        for strategy in sorted(best_df['strategy'].unique()):
            strategy_df = best_df[best_df['strategy'] == strategy].copy()
            strategy_df = strategy_df.sort_index()
            strategy_df['cum_pnl'] = strategy_df['pnl'].cumsum()
            ax7.plot(range(len(strategy_df)), strategy_df['cum_pnl'], 
                    label=f'{strategy} min', linewidth=2)
        ax7.set_xlabel('Bet Number')
        ax7.set_ylabel('Cumulative P&L (£)')
        ax7.set_title('Cumulative P&L by Strategy')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 8. Kelly Stake Distribution
        ax8 = plt.subplot(3, 3, 8)
        best_df['stake'].hist(bins=30, ax=ax8, alpha=0.7, color='green')
        ax8.axvline(STAKE_PER_BET, color='red', linestyle='--', 
                   label=f"Base Stake: £{STAKE_PER_BET}")
        ax8.set_xlabel('Stake Size (£)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Distribution of Kelly Stakes')
        ax8.legend()
        
        # 9. Model Agreement Analysis
        ax9 = plt.subplot(3, 3, 9)
        agreement_df = df_all.groupby(['conf_threshold', 'value_threshold']).agg({
            'pnl': 'sum',
            'stake': 'count'
        })
        
        # Create bubble plot
        for idx, row in agreement_df.iterrows():
            ax9.scatter(idx[0], idx[1], s=row['stake']*10, 
                       c=[row['pnl']], cmap='RdYlGn', 
                       vmin=-1000, vmax=1000, alpha=0.6)
        
        ax9.set_xlabel('Confidence Threshold')
        ax9.set_ylabel('Value Threshold')
        ax9.set_title('P&L by Threshold (size = # bets)')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                                   norm=plt.Normalize(vmin=-1000, vmax=1000))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax9)
        cbar.set_label('Total P&L (£)')
        
        plt.tight_layout()
        plt.savefig(self.stax_model.output_dir / 'comprehensive_backtest_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Enhanced Stax Meta-Model Pipeline')
    parser.add_argument('--mode', choices=['train', 'backtest', 'all'], 
                       default='all', help='Operation mode')
    parser.add_argument('--strategies', nargs='+', type=int, 
                       default=DEFAULT_STRATEGIES,
                       help='Betting time strategies in minutes')
    parser.add_argument('--conf_thresholds', nargs='+', type=float,
                       default=[0.55, 0.60, 0.65, 0.70],
                       help='Confidence thresholds to test')
    parser.add_argument('--value_thresholds', nargs='+', type=float,
                       default=[0.00, 0.05, 0.10, 0.15],
                       help='Value thresholds to test')
    
    args = parser.parse_args()
    
    # Initialize Enhanced Stax model
    stax = EnhancedStaxModel()
    
    if args.mode in ['train', 'all']:
        print("=== Training Enhanced Stax Meta-Model ===")
        
        # Generate enhanced meta-features
        meta_df = stax.generate_enhanced_meta_features(DATA_DIR / "Training")
        
        if meta_df.empty:
            print("No meta-features generated. Exiting.")
            return
        
        print(f"Generated {len(meta_df)} enhanced samples for training")
        
        # Prepare data
        X_meta = meta_df.drop('final_outcome', axis=1)
        y_meta = meta_df['final_outcome']
        
        # Train multiple meta-models
        stax.train_meta_models(meta_df, y_meta)
    
    if args.mode in ['backtest', 'all']:
        # Load trained models if only backtesting
        if args.mode == 'backtest':
            stax.load_saved_models()
        
        print("\n=== Running Advanced Backtest ===")
        backtester = ValueBettingBacktester(stax)
        
        # Run backtest with multiple threshold configurations
        backtester.run_advanced_backtest(
            DATA_DIR / "Backtest", 
            args.strategies,
            args.conf_thresholds,
            args.value_thresholds
        )
        
        # Comprehensive analysis
        best_config, summary = backtester.analyze_comprehensive_results()
        
        print(f"\n✅ Analysis complete! Results saved to {stax.output_dir}")
        print(f"📊 Best configuration: Confidence>{best_config[0]:.2f}, Value>{best_config[1]:.2f}")
        print(f"💰 Best ROI: {summary.loc[best_config, 'roi']:.2f}%")
    
    print("\n--- Enhanced Stax Meta-Model Pipeline Complete ---")


if __name__ == '__main__':
    main()