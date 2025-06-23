# scripts/train_lstm_seq5.py
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import argparse

warnings.filterwarnings('ignore')


def get_team_names(match_name: str, odds_data: dict):
    if ' vs ' in match_name:
        teams = match_name.split(' vs ')
        return teams[0].strip(), teams[1].strip()
    all_teams = {k.strip() for d in odds_data.values() for k in d if isinstance(k, str) and k.lower() != 'draw'}
    sorted_teams = sorted(all_teams)
    if len(sorted_teams) >= 2:
        return sorted_teams[0], sorted_teams[1]
    return None, None


class MatchOddsLSTM:
    def __init__(self, sequence_length=5, features=4):
        self.sequence_length = sequence_length
        self.features = features
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

    def load_match_data(self, json_path):
        data = json.load(open(json_path, 'r'))[:180]
        final_home, final_away = map(int, data[-1]['score'].split(' - '))
        outcome = 0 if final_home > final_away else 1 if final_away > final_home else 2
        feats = []
        home_team, away_team = get_team_names(
            data[0].get('match', ''), data[0].get('odds', {})
        )
        for entry in data:
            parts = entry['score'].split(' - ')
            hd, ad = int(parts[0]), int(parts[1])
            score_diff = hd - ad
            h_odds, a_odds, d_odds = [], [], []
            for od in entry.get('odds', {}).values():
                keys = list(od.keys())
                teams = [k for k in keys if k != 'Draw']
                if len(teams) >= 2:
                    h_odds.append(od[teams[0]]); a_odds.append(od[teams[1]])
                if 'Draw' in od:
                    d_odds.append(od['Draw'])
            feats.append([
                np.mean(h_odds) if h_odds else 2.0,
                np.mean(a_odds) if a_odds else 2.0,
                np.mean(d_odds) if d_odds else 3.0,
                score_diff
            ])
        return np.array(feats), outcome

    def create_sequences(self, features, outcome):
        seqs, labs = [], []
        for i in range(self.sequence_length, len(features)):
            seqs.append(features[i-self.sequence_length:i])
            labs.append(outcome)
        return seqs, labs

    def load_all(self, data_dir):
        data_path = Path(data_dir)
        json_files = list(data_path.glob('*.json')) + list(data_path.glob('*.JSON'))
        X, y = [], []
        for f in json_files:
            feats, out = self.load_match_data(str(f))
            if len(feats) < self.sequence_length: continue
            seqs, labs = self.create_sequences(feats, out)
            X.extend(seqs); y.extend(labs)
        X = np.array(X);
        y = np.array(y)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=3)
        return X, y_cat

    def build_model(self):
        m = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, self.features)),
            Dropout(0.2), BatchNormalization(),
            LSTM(50, return_sequences=False), Dropout(0.2), BatchNormalization(),
            Dense(25, activation='relu'), Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        m.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = m
        return m

    def train(self, X, y, val_split=0.2, batch_size=64, epochs=30, patience=10):
        flat = X.reshape(-1, self.features)
        flat_norm = self.scaler.fit_transform(flat)
        Xn = flat_norm.reshape(-1, self.sequence_length, self.features)
        X_train, X_val, y_train, y_val = train_test_split(
            Xn, y, test_size=val_split, random_state=42, stratify=y.argmax(axis=1)
        )
        cb = [
            EarlyStopping(patience=patience, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint('best_seq5.h5', save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5, monitor='val_loss')
        ]
        history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=1
        )
        self.is_fitted = True
        return history

    def save(self, out_dir):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.model.save(out_path / 'lstm_seq5.h5')
        joblib.dump(self.scaler, out_path / 'scaler_seq5.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM Seq5 Model')
    parser.add_argument('--data_dir', type=str, default=str(
        Path(__file__).resolve().parent.parent / 'data' / 'Training'
    ), help='Path to training JSON files')
    parser.add_argument('--model_dir', type=str, default=str(
        Path(__file__).resolve().parent.parent / 'models' / 'lstm_seq5'
    ), help='Directory to save trained model and scaler')
    args = parser.parse_args()

    print(f"Loading sequences from {args.data_dir}")
    lstm = MatchOddsLSTM(sequence_length=5, features=4)
    lstm.build_model()
    X, y = lstm.load_all(args.data_dir)
    print(f"Loaded {len(X)} sequences for seq_len=5")
    history = lstm.train(X, y)
    lstm.save(args.model_dir)
    print(f"Training complete. Model and scaler saved to '{args.model_dir}'.")