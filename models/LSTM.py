import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MatchOddsLSTM:
    def __init__(self, sequence_length: int = 10, features: int = 4):
        """
        Initialize the LSTM model for match outcome prediction.
        
        Args:
            sequence_length: Number of time steps to look back
            features: Number of features per time step
        """
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_match_data(self, json_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and process a single match JSON file.
        
        Returns:
            features: Array of shape (n_intervals, 4) with [home_odds, away_odds, draw_odds, score_diff]
            outcome: 0=home win, 1=away win, 2=draw
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Limit to first 180 intervals (full match + stoppage time)
        data = data[:180]
        
        features = []
        for entry in data:
            # Extract score
            score_parts = entry['score'].split(' - ')
            home_score = int(score_parts[0])
            away_score = int(score_parts[1])
            score_diff = home_score - away_score
            
            # Calculate average odds across all bookmakers
            odds_dict = entry['odds']
            home_odds = []
            away_odds = []
            draw_odds = []
            
            for bookmaker, odds in odds_dict.items():
                # Get all keys from odds dictionary
                odds_keys = list(odds.keys())
                
                # Typically: first key is home team, second is away team, 'Draw' is draw
                # Filter out 'Draw' to find team names
                team_keys = [k for k in odds_keys if k != 'Draw']
                
                if len(team_keys) >= 2:
                    # Assume first team is home, second is away
                    home_odds.append(odds[team_keys[0]])
                    away_odds.append(odds[team_keys[1]])
                elif len(team_keys) == 1:
                    # Sometimes only one team odds might be present
                    home_odds.append(odds[team_keys[0]])
                    
                if 'Draw' in odds:
                    draw_odds.append(odds['Draw'])
            
            # Average odds
            avg_home_odds = np.mean(home_odds) if home_odds else 2.0
            avg_away_odds = np.mean(away_odds) if away_odds else 2.0
            avg_draw_odds = np.mean(draw_odds) if draw_odds else 3.0
            
            features.append([avg_home_odds, avg_away_odds, avg_draw_odds, score_diff])
        
        # Determine match outcome from final score
        final_entry = data[-1]
        final_score = final_entry['score'].split(' - ')
        final_home = int(final_score[0])
        final_away = int(final_score[1])
        
        if final_home > final_away:
            outcome = 0  # Home win
        elif final_away > final_home:
            outcome = 1  # Away win
        else:
            outcome = 2  # Draw
            
        return np.array(features), outcome
    
    def create_sequences(self, features: np.ndarray, outcome: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create overlapping sequences from match data.
        
        Returns:
            sequences: List of arrays with shape (sequence_length, features)
            labels: List of outcomes
        """
        sequences = []
        labels = []
        
        # Create sequences starting from different points in the match
        for i in range(self.sequence_length, len(features)):
            sequence = features[i-self.sequence_length:i]
            sequences.append(sequence)
            labels.append(outcome)
            
        return sequences, labels
    
    def load_all_matches(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all match data from the training directory.
        """
        data_path = Path(data_dir)
        
        # Debug: Check if path exists
        if not data_path.exists():
            print(f"ERROR: Directory does not exist: {data_path}")
            print(f"Current working directory: {Path.cwd()}")
            return np.array([]), np.array([])
        
        # Try both lowercase and uppercase extensions
        json_files = list(data_path.glob('*.json')) + list(data_path.glob('*.JSON'))
        
        # Debug: Show what files are in the directory
        all_files = list(data_path.iterdir())
        print(f"Directory contents: {data_path}")
        print(f"Total files in directory: {len(all_files)}")
        if len(all_files) > 0:
            print(f"First few files: {[f.name for f in all_files[:5]]}")
        
        print(f"Found {len(json_files)} match files")
        
        all_sequences = []
        all_labels = []
        
        for i, json_file in enumerate(json_files):
            try:
                features, outcome = self.load_match_data(str(json_file))
                
                # Skip matches with too few intervals
                if len(features) < self.sequence_length:
                    continue
                    
                sequences, labels = self.create_sequences(features, outcome)
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(json_files)} matches")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # Convert to numpy arrays
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        # One-hot encode labels
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)
        
        print(f"Total sequences: {len(X)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Label distribution: Home wins: {np.sum(y==0)}, Away wins: {np.sum(y==1)}, Draws: {np.sum(y==2)}")
        
        return X, y_onehot
    
    def build_model(self):
        """
        Build the LSTM model architecture.
        """
        model = Sequential([
            # First LSTM layer
            LSTM(100, return_sequences=True, 
                 input_shape=(self.sequence_length, self.features)),
            Dropout(0.2),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2,
              batch_size: int = 32,
              epochs: int = 50,
              patience: int = 10):
        """
        Train the LSTM model.
        """
        # Normalize features
        X_reshaped = X.reshape(-1, self.features)
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(-1, self.sequence_length, self.features)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001, monitor='val_loss')
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        return history
    
    def predict_match(self, features: np.ndarray) -> np.ndarray:
        """
        Predict outcome probabilities for a match.
        
        Args:
            features: Array of shape (n_intervals, 4)
            
        Returns:
            probabilities: Array of shape (n_predictions, 3) with [P_home, P_away, P_draw]
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        
        # Make predictions for each valid sequence
        for i in range(self.sequence_length, len(features)):
            sequence = features[i-self.sequence_length:i]
            
            # Normalize
            sequence_normalized = self.scaler.transform(sequence.reshape(-1, self.features))
            sequence_normalized = sequence_normalized.reshape(1, self.sequence_length, self.features)
            
            # Predict
            pred = self.model.predict(sequence_normalized, verbose=0)
            predictions.append(pred[0])
        
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save the trained model and scaler."""
        import joblib
        self.model.save(f"{path}/lstm_model.h5")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        
    def load_saved_model(self, path: str):
        """Load a previously trained model and scaler."""
        import joblib
        self.model = load_model(f"{path}/lstm_model.h5")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.is_fitted = True


# Example usage and training script
if __name__ == "__main__":
    # Initialize model with different sequence lengths to experiment
    sequence_lengths = [5, 10, 20]  # 200s, 400s, 800s of history
    
    for seq_len in sequence_lengths:
        print(f"\n{'='*50}")
        print(f"Training model with sequence length: {seq_len}")
        print(f"{'='*50}\n")
        
        # Create model
        lstm_model = MatchOddsLSTM(sequence_length=seq_len, features=4)
        
        # Build architecture
        lstm_model.build_model()
        print(lstm_model.model.summary())
        
        # Load training data
        X, y = lstm_model.load_all_matches("../data/Training")
        
        # Train model
        history = lstm_model.train(X, y, epochs=30, batch_size=64)
        
        # Save model
        lstm_model.save_model(f"./lstm_seq{seq_len}")
        
        # Print final performance
        val_loss = min(history.history['val_loss'])
        val_acc = max(history.history['val_accuracy'])
        print(f"\nBest validation loss: {val_loss:.4f}")
        print(f"Best validation accuracy: {val_acc:.4f}")