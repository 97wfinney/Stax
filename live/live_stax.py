#!/usr/bin/env python3
"""
Live Stax Model Web Application
Provides real-time predictions for EPL/EFL matches
"""

import json
import time
import threading
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import warnings


warnings.filterwarnings('ignore')

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models" / "stax3"
DATA_DIR = ROOT_DIR / "data" / "live"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
API_KEY = '2cfedcda4335d5c8721d3d556c841128'
TEST_API_KEY = '2778c9e338a6fd0e6e7dee4e5cd34557'  # Test API key for League 1
REGIONS = 'uk'
MARKETS = 'h2h'
ODDS_FORMAT = 'decimal'
DATE_FORMAT = 'iso'

# Model features (from backtest.py)
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

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class LiveStaxPredictor:
    def __init__(self):
        self.current_bank = 100.0
        self.kelly_fraction = 0.25
        self.confidence_threshold = 0.65
        self.leagues_to_monitor = set()
        self.live_matches = {}
        self.predictions = []
        self.executed_bets = []
        self.is_running = False
        self.current_matches_data = {}  # Add this
        self._load_models()
        
    def _load_models(self):
        """Load all required models"""
        print("Loading models...")
        
        # Load config
        config_path = MODELS_DIR / 'stax_config.joblib'
        cfg = joblib.load(config_path)
        self.team_to_idx = cfg.get('team_to_idx', {})
        self.league_to_idx = cfg.get('league_to_idx', {})
        self.weights = cfg.get('model_weights', [0.34, 0.33, 0.33])
        
        # Load base models
        self.models = {}
        self.scalers = {}
        
        # Logistic Regression
        lr_dir = MODELS_DIR / 'logistic_regression'
        self.models['lr'] = joblib.load(lr_dir / 'model.joblib')
        self.scalers['lr'] = joblib.load(lr_dir / 'scaler.joblib')
        
        # XGBoost
        xgb_dir = MODELS_DIR / 'xgboost'
        self.models['xgb'] = joblib.load(xgb_dir / 'model.joblib')
        self.scalers['xgb'] = joblib.load(xgb_dir / 'scaler.joblib')
        
        # LSTM
        lstm_dir = MODELS_DIR / 'lstm'
        self.models['lstm'] = tf.keras.models.load_model(str(lstm_dir / 'model.h5'))
        self.scalers['lstm'] = joblib.load(lstm_dir / 'scaler.pkl')
        
        # Meta model
        meta_model_path = MODELS_DIR / 'meta_model_simple_nn.h5'
        if not meta_model_path.exists():
            meta_model_path = MODELS_DIR / 'meta_model_nn.h5'
        self.meta_model = tf.keras.models.load_model(str(meta_model_path))
        self.meta_scaler = joblib.load(MODELS_DIR / 'meta_scaler_standard.joblib')
        
        # Load any previously executed bets
        bets_file = DATA_DIR / 'executed_bets.json'
        if bets_file.exists():
            with open(bets_file, 'r') as f:
                self.executed_bets = json.load(f)
                print(f"Loaded {len(self.executed_bets)} previous bets")
        else:
            print("No previous bets found, starting fresh")
        
        print("Models loaded successfully!")
    
    def fetch_odds(self, sport):
        """Fetch odds data from API"""
        # Use test API key for League 1
        api_key = TEST_API_KEY if sport == 'soccer_england_league1' else API_KEY
        
        odds_url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
        params = {
            'api_key': api_key,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
        }
        try:
            response = requests.get(odds_url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching odds: {e}")
        return []
    
    def fetch_scores(self, sport):
        """Fetch scores data from API"""
        # Use test API key for League 1
        api_key = TEST_API_KEY if sport == 'soccer_england_league1' else API_KEY
        
        scores_url = f'https://api.the-odds-api.com/v4/sports/{sport}/scores/'
        params = {
            'apiKey': api_key,
            'dateFormat': DATE_FORMAT,
        }
        try:
            response = requests.get(scores_url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching scores: {e}")
        return []
    
    def process_match_data(self, match_id, match_data):
        """Process match data for prediction"""
        if len(match_data) < 2:
            return None, None
        
        # Extract team names
        first_entry = match_data[0]
        home_team, away_team = first_entry['match'].split(' vs ')
        
        # Build dataframe
        rows = []
        for i, entry in enumerate(match_data):
            score_parts = entry['score'].split(' - ')
            if len(score_parts) != 2:
                continue
                
            home_score, away_score = int(score_parts[0]), int(score_parts[1])
            
            # Extract odds
            home_odds, away_odds, draw_odds = [], [], []
            for bookmaker, odds in entry['odds'].items():
                if home_team in odds:
                    home_odds.append(odds[home_team])
                if away_team in odds:
                    away_odds.append(odds[away_team])
                if 'Draw' in odds:
                    draw_odds.append(odds['Draw'])
            
            if home_odds and away_odds and draw_odds:
                rows.append({
                    'time_elapsed_s': i * 40,
                    'home_score': home_score,
                    'away_score': away_score,
                    'avg_home_odds': np.mean(home_odds),
                    'avg_away_odds': np.mean(away_odds),
                    'avg_draw_odds': np.mean(draw_odds)
                })
        
        if not rows:
            return None, None
            
        df = pd.DataFrame(rows)
        
        # Add derived features
        df['score_diff'] = df['home_score'] - df['away_score']
        
        # Rolling features
        for col in ['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']:
            key = col.split('_')[1]
            df[f'std_{key}_odds'] = df[col].rolling(5).std().fillna(0)
            df[f'{key}_odds_momentum'] = df[col].diff().rolling(5).mean().fillna(0)
        
        df['prob_home'] = 1 / df['avg_home_odds']
        df['prob_away'] = 1 / df['avg_away_odds']
        df['prob_draw'] = 1 / df['avg_draw_odds']
        
        return df, match_data
    
    def predict_probabilities(self, df, match_data):
        """Generate predictions using the stax model"""
        if df is None or len(df) < SEQUENCE_LENGTH:
            return None
            
        # Get predictions from base models
        lr_p = self.models['lr'].predict_proba(
            self.scalers['lr'].transform(df[LR_FEATURES])
        )
        xgb_p = self.models['xgb'].predict_proba(
            self.scalers['xgb'].transform(df[XGB_FEATURES])
        )
        
        # LSTM sequences
        seqs = []
        for i in range(len(df)):
            seq = df[LSTM_FEATURES].iloc[max(0, i-SEQUENCE_LENGTH+1):i+1].values
            if len(seq) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH-len(seq), seq.shape[1]))
                seq = np.vstack([pad, seq])
            seqs.append(seq)
        
        arr = np.array(seqs)
        flat = arr.reshape(-1, arr.shape[-1])
        scaled = self.scalers['lstm'].transform(flat).reshape(arr.shape)
        lstm_p = self.models['lstm'].predict(scaled, verbose=0)
        
        # Build meta features
        meta_features_list = []
        for i in range(df.shape[0]):
            p_lr_H, p_lr_A, p_lr_D = lr_p[i]
            p_xgb_H, p_xgb_A, p_xgb_D = xgb_p[i]
            p_lstm_H, p_lstm_A, p_lstm_D = lstm_p[i]
            
            avg_H = (p_lr_H + p_xgb_H + p_lstm_H) / 3
            avg_A = (p_lr_A + p_xgb_A + p_lstm_A) / 3
            avg_D = (p_lr_D + p_xgb_D + p_lstm_D) / 3
            
            # Disagreement metrics
            preds = np.array([
                [p_lr_H, p_lr_A, p_lr_D],
                [p_xgb_H, p_xgb_A, p_xgb_D],
                [p_lstm_H, p_lstm_A, p_lstm_D]
            ])
            avg_pred = preds.mean(axis=0)
            entropy = -np.sum(avg_pred * np.log(avg_pred + 1e-10))
            std_dev = preds.std(axis=0).mean()
            
            time_factor = min(i / 90, 1.0)
            odds_vals = df[['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].iloc[i].values
            market_overround = (1 / odds_vals).sum() - 1
            score_diff = df['score_diff'].iloc[i]
            
            row_feats = [
                p_lr_H, p_lr_A, p_lr_D,
                p_xgb_H, p_xgb_A, p_xgb_D,
                p_lstm_H, p_lstm_A, p_lstm_D,
                avg_H, avg_A, avg_D,
                entropy, std_dev, 0.0,  # max_diff placeholder
                time_factor, market_overround, score_diff
            ]
            meta_features_list.append(row_feats)
        
        meta_features = np.array(meta_features_list)
        scaled_meta = self.meta_scaler.transform(meta_features)
        
        # Get ensemble predictions
        ensemble_probs = self.meta_model.predict(scaled_meta, verbose=0)
        
        return ensemble_probs
    
    def calculate_kelly_stake(self, prob, odds):
        """Calculate stake using Kelly criterion"""
        edge = prob * odds - 1
        if edge <= 0:
            return 0
        
        kelly_stake = (edge / (odds - 1)) * self.kelly_fraction
        max_stake = self.current_bank * 0.05  # Max 5% of bank
        
        return min(kelly_stake * self.current_bank, max_stake)
    
    def update_loop(self):
        """Main update loop that runs every 40 seconds"""
        while self.is_running:
            try:
                all_matches = {}
                
                # Fetch data for selected leagues
                for league in self.leagues_to_monitor:
                    if league == 'EPL':
                        sport = 'soccer_epl'
                    elif league == 'EFL':
                        sport = 'soccer_efl_champ'
                    elif league == 'L1TEST':
                        sport = 'soccer_england_league1'
                    else:
                        continue
                    
                    odds_data = self.fetch_odds(sport)
                    scores_data = self.fetch_scores(sport)
                    
                    # Process live matches
                    for score in scores_data:
                        if score.get('scores') and not score.get('completed'):
                            match_id = score['id']
                            event = next((item for item in odds_data if item['id'] == match_id), None)
                            
                            if event:
                                # Get odds from first 7 bookmakers
                                bookmakers_odds = event.get('bookmakers', [])[:7]
                                odds_info = {}
                                
                                for bm in bookmakers_odds:
                                    for market in bm.get('markets', []):
                                        if market['key'] == 'h2h':
                                            odds_info[bm['title']] = {
                                                outcome['name']: outcome['price'] 
                                                for outcome in market['outcomes']
                                            }
                                
                                home_score = next((s['score'] for s in score['scores'] 
                                                 if s['name'] == event['home_team']), '0')
                                away_score = next((s['score'] for s in score['scores'] 
                                                 if s['name'] == event['away_team']), '0')
                                
                                match_entry = {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "match": f"{event['home_team']} vs {event['away_team']}",
                                    "score": f"{home_score} - {away_score}",
                                    "odds": odds_info,
                                    "league": league
                                }
                                
                                # Create unique match key based on team names
                                match_key = f"{event['home_team']}_vs_{event['away_team']}".replace(' ', '_')
                                
                                # Update match history
                                if match_key not in self.live_matches:
                                    self.live_matches[match_key] = []
                                    print(f"New match detected: {match_entry['match']}")
                                
                                self.live_matches[match_key].append(match_entry)
                                all_matches[match_key] = self.live_matches[match_key]
                                print(f"Match history length for {match_key}: {len(self.live_matches[match_key])}")
                
                # Save complete match histories periodically (every 5 updates)
                for match_key, match_history in self.live_matches.items():
                    if len(match_history) % 5 == 0 and len(match_history) > 0:
                        # Save accumulated history
                        history_file = DATA_DIR / f"{match_key}_history.json"
                        with open(history_file, 'w') as f:
                            json.dump(match_history, f, indent=2)
                        print(f"Saved {match_key} history: {len(match_history)} entries")
                
                # Clean up completed matches from live_matches to save memory
                completed_matches = []
                for match_key, match_history in self.live_matches.items():
                    if len(match_history) > 0:
                        latest_entry = match_history[-1]
                        last_update = datetime.fromisoformat(latest_entry['timestamp'].replace('Z', '+00:00'))
                        time_since_update = (datetime.now(timezone.utc) - last_update).seconds
                        
                        # If no update for 15 minutes, consider match completed
                        if time_since_update > 900:
                            completed_matches.append(match_key)
                
                # Remove completed matches
                for match_key in completed_matches:
                    print(f"Match completed: {self.live_matches[match_key][-1]['match']}")
                    del self.live_matches[match_key]
                
                # Generate predictions for each match
                new_predictions = []
                for match_id, match_history in all_matches.items():
                    df, processed_data = self.process_match_data(match_id, match_history)
                    
                    if df is not None and len(df) >= SEQUENCE_LENGTH:
                        probs = self.predict_probabilities(df, processed_data)
                        
                        if probs is not None:
                            latest_probs = probs[-1]
                            latest_data = match_history[-1]
                            latest_df_row = df.iloc[-1]
                            
                            # Calculate confidence and determine bet
                            confidence = float(latest_probs.max())
                            pred_idx = int(latest_probs.argmax())
                            pred_outcome = ['Home', 'Away', 'Draw'][pred_idx]
                            
                            # Get average odds
                            odds_list = [
                                latest_df_row['avg_home_odds'],
                                latest_df_row['avg_away_odds'],
                                latest_df_row['avg_draw_odds']
                            ]
                            selected_odds = odds_list[pred_idx]
                            
                            # Calculate stake
                            stake = 0
                            if confidence >= self.confidence_threshold:
                                stake = self.calculate_kelly_stake(confidence, selected_odds)
                            
                            prediction = {
                                'match_id': match_key,  # Use match_key instead of match_id
                                'match': latest_data['match'],
                                'score': latest_data['score'],
                                'league': latest_data['league'],
                                'prediction': pred_outcome,
                                'confidence': confidence,
                                'probabilities': {
                                    'home': float(latest_probs[0]),
                                    'away': float(latest_probs[1]),
                                    'draw': float(latest_probs[2])
                                },
                                'odds': {
                                    'home': float(odds_list[0]),
                                    'away': float(odds_list[1]),
                                    'draw': float(odds_list[2])
                                },
                                'selected_odds': selected_odds,
                                'recommended_stake': stake,
                                'timestamp': datetime.now().isoformat(),
                                'minute': len(match_history) * 40 // 60
                            }
                            
                            new_predictions.append(prediction)
                
                # Update predictions list (keep last 50)
                self.predictions = new_predictions 
                
                # Store the data for API access
                self.current_matches_data = {
                    'matches': {k: v for k, v in all_matches.items()},
                    'predictions': self.predictions[:20],
                    'bank': self.current_bank,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Emit updates via WebSocket
                print(f"Emitting update: {len(all_matches)} matches, {len(self.predictions)} predictions")
                try:
                    socketio.emit('update', self.current_matches_data)
                except Exception as e:
                    print(f"Error emitting update: {e}")
                
            except Exception as e:
                print(f"Error in update loop: {e}")
            
            # Check for completed matches and update bet results
            self.update_bet_results()
            
            time.sleep(40)
    
    def execute_bet(self, prediction_data):
        """Execute a bet and save to tracking file"""
        bet_record = {
            **prediction_data,
            'executed_at': datetime.now().isoformat(),
            'bank_at_bet': self.current_bank,
            'status': 'pending'
        }
        
        self.executed_bets.append(bet_record)
        
        # Save to file
        bets_file = DATA_DIR / 'executed_bets.json'
        with open(bets_file, 'w') as f:
            json.dump(self.executed_bets, f, indent=2)
        
        return bet_record
    
    def update_bet_results(self):
        """Update results of completed matches"""
        if not self.executed_bets:
            return
            
        updated = False
        
        for bet in self.executed_bets:
            if bet['status'] != 'pending':
                continue
                
            match_id = bet['match_id']
            
            # Check if we have this match in our live data
            if match_id in self.live_matches:
                match_data = self.live_matches[match_id]
                latest_entry = match_data[-1]
                
                # Check if match might be finished (no updates for 10+ minutes)
                last_update = datetime.fromisoformat(latest_entry['timestamp'].replace('Z', '+00:00'))
                time_since_update = (datetime.now(timezone.utc) - last_update).seconds
                
                if time_since_update > 600:  # 10 minutes
                    # Get final score
                    final_score = latest_entry['score'].split(' - ')
                    home_score = int(final_score[0])
                    away_score = int(final_score[1])
                    
                    # Determine actual outcome
                    if home_score > away_score:
                        actual_outcome = 'Home'
                    elif away_score > home_score:
                        actual_outcome = 'Away'
                    else:
                        actual_outcome = 'Draw'
                    
                    # Update bet status
                    if bet['prediction'] == actual_outcome:
                        bet['status'] = 'won'
                        bet['pnl'] = bet['recommended_stake'] * (bet['selected_odds'] - 1)
                    else:
                        bet['status'] = 'lost'
                        bet['pnl'] = -bet['recommended_stake']
                    
                    bet['final_score'] = latest_entry['score']
                    bet['actual_outcome'] = actual_outcome
                    updated = True
        
        # Save if any updates were made
        if updated:
            bets_file = DATA_DIR / 'executed_bets.json'
            with open(bets_file, 'w') as f:
                json.dump(self.executed_bets, f, indent=2)
            
            # Emit update to frontend
            socketio.emit('bets_updated', self.executed_bets)

# Initialize predictor
predictor = LiveStaxPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    data = request.json
    predictor.leagues_to_monitor = set(data.get('leagues', []))
    predictor.confidence_threshold = data.get('confidence_threshold', 0.65)
    predictor.kelly_fraction = data.get('kelly_fraction', 0.25)
    
    if not predictor.is_running:
        predictor.is_running = True
        thread = threading.Thread(target=predictor.update_loop)
        thread.daemon = True
        thread.start()
    
    return jsonify({'status': 'started', 'leagues': list(predictor.leagues_to_monitor)})

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    predictor.is_running = False
    return jsonify({'status': 'stopped'})

@app.route('/api/update_bank', methods=['POST'])
def update_bank():
    data = request.json
    predictor.current_bank = float(data.get('bank', 100))
    return jsonify({'bank': predictor.current_bank})

@app.route('/api/execute_bet', methods=['POST'])
def execute_bet():
    prediction_data = request.json
    bet_record = predictor.execute_bet(prediction_data)
    return jsonify(bet_record)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current monitoring status and statistics"""
    active_matches = len(predictor.live_matches)
    total_predictions = len(predictor.predictions)
    pending_bets = len([b for b in predictor.executed_bets if b['status'] == 'pending'])
    won_bets = len([b for b in predictor.executed_bets if b['status'] == 'won'])
    lost_bets = len([b for b in predictor.executed_bets if b['status'] == 'lost'])
    
    return jsonify({
        'is_running': predictor.is_running,
        'leagues': list(predictor.leagues_to_monitor),
        'active_matches': active_matches,
        'total_predictions': total_predictions,
        'pending_bets': pending_bets,
        'won_bets': won_bets,
        'lost_bets': lost_bets,
        'current_bank': predictor.current_bank
    })

@app.route('/api/get_bets', methods=['GET'])
def get_bets():
    return jsonify(predictor.executed_bets)

@app.route('/api/get_current_state', methods=['GET'])
def get_current_state():
    """Get current state of matches and predictions"""
    # Return the stored data from the predictor
    if predictor.current_matches_data:
        return jsonify(predictor.current_matches_data)
    else:
        return jsonify({
            'matches': {k: v for k, v in predictor.live_matches.items()},
            'predictions': predictor.predictions[:20],
            'bank': predictor.current_bank,
            'is_running': predictor.is_running
        })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to Live Stax Model'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("Starting Live Stax Model Server...")
    print(f"Data will be saved to: {DATA_DIR}")
    socketio.run(app, debug=False, host='0.0.0.0', port=5002)  # Set debug=False