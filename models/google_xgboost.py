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
from tqdm import tqdm

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# --- 1. CONFIGURATION ---
TRAINING_DIR = Path("../data/Training")
MODEL_OUTPUT_DIR = Path("./xgboost_model")
MOMENTUM_WINDOW = 5 # Number of intervals (e.g., 5 * 40s = ~3.3 mins) to calculate momentum over.


def get_team_names(match_name: str, odds_keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Robustly determines home and away team names from match data."""
    if ' vs ' in match_name:
        teams = match_name.split(' vs ')
        return teams[0].strip(), teams[1].strip()
    
    non_draw_keys = [k for k in odds_keys if k.lower() != 'draw']
    if len(non_draw_keys) >= 2:
        return non_draw_keys[0], non_draw_keys[1]
    return None, None


def process_match_file(json_path: Path) -> Optional[pd.DataFrame]:
    """
    Loads a single match JSON, engineers features, and returns a clean DataFrame.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        return None

    # --- Basic Info and Final Outcome ---
    match_name = data[0].get('match', 'Unknown Match')
    final_score_str = data[-1]['score']
    final_home, final_away = map(int, final_score_str.split(' - '))
    if final_home > final_away:
        final_outcome = 0
    elif final_away > final_home:
        final_outcome = 1
    else:
        final_outcome = 2

    home_team, away_team = get_team_names(match_name, list(data[0]['odds'].values())[0].keys())
    if not home_team:
        return None
    
    # --- Feature Engineering per Interval ---
    rows = []
    for i, entry in enumerate(data):
        home_score, away_score = map(int, entry['score'].split(' - '))
        
        h_odds, a_odds, d_odds = [], [], []
        for bookmaker_odds in entry['odds'].values():
            h_odds.append(bookmaker_odds.get(home_team))
            a_odds.append(bookmaker_odds.get(away_team))
            d_odds.append(bookmaker_odds.get('Draw'))
        
        # Filter out None values in case a bookmaker is missing
        h_odds = [o for o in h_odds if o is not None]
        a_odds = [o for o in a_odds if o is not None]
        d_odds = [o for o in d_odds if o is not None]

        rows.append({
            'time_elapsed_s': i * 40,
            'home_score': home_score,
            'away_score': away_score,
            'score_diff': home_score - away_score,
            'avg_home_odds': np.mean(h_odds) if h_odds else np.nan,
            'avg_away_odds': np.mean(a_odds) if a_odds else np.nan,
            'avg_draw_odds': np.mean(d_odds) if d_odds else np.nan,
            'std_home_odds': np.std(h_odds) if len(h_odds) > 1 else 0,
            'std_away_odds': np.std(a_odds) if len(a_odds) > 1 else 0,
            'std_draw_odds': np.std(d_odds) if len(d_odds) > 1 else 0,
        })

    df = pd.DataFrame(rows)

    # --- Advanced Rolling Features ---
    df['home_odds_momentum'] = df['avg_home_odds'].diff().rolling(window=MOMENTUM_WINDOW).mean()
    df['away_odds_momentum'] = df['avg_away_odds'].diff().rolling(window=MOMENTUM_WINDOW).mean()
    df['draw_odds_momentum'] = df['avg_draw_odds'].diff().rolling(window=MOMENTUM_WINDOW).mean()

    # --- Implied Probabilities ---
    df['prob_home'] = 1 / df['avg_home_odds']
    df['prob_away'] = 1 / df['avg_away_odds']
    df['prob_draw'] = 1 / df['avg_draw_odds']

    # --- Add Target Variable ---
    df['final_outcome'] = final_outcome
    
    return df


def create_training_data(training_dir: Path) -> pd.DataFrame:
    """
    Processes all JSON files in the training directory into a single master DataFrame.
    """
    json_files = list(training_dir.glob('*.json'))
    print(f"Found {len(json_files)} match files in {training_dir}.")
    
    all_match_dfs = []
    for file_path in tqdm(json_files, desc="Processing training files"):
        match_df = process_match_file(file_path)
        if match_df is not None:
            all_match_dfs.append(match_df)
            
    print("Combining all matches into a single dataset...")
    master_df = pd.concat(all_match_dfs, ignore_index=True)
    
    # Drop rows with NaN values resulting from rolling calculations
    master_df.dropna(inplace=True)
    
    print(f"Master dataset created with {len(master_df)} samples.")
    return master_df

def train_model(df: pd.DataFrame) -> Tuple[xgb.XGBClassifier, StandardScaler]:
    """
    Trains an XGBoost model on the provided DataFrame and returns the model and scaler.
    """
    # --- Define Features and Target ---
    features = [col for col in df.columns if col != 'final_outcome']
    target = 'final_outcome'
    
    X = df[features]
    y = df[target]
    
    # --- Train/Validation Split ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Training XGBoost model on {len(X_train)} samples...")
    
    # --- XGBoost Model Training (MODIFIED) ---
    # We move 'early_stopping_rounds' into the constructor here.
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=50, # <-- MOVED TO HERE
        random_state=42
    )
    
    # The .fit() call now only needs the data and the evaluation set.
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=100
    )
    
    # --- Evaluation ---
    print("\n--- Model Evaluation on Validation Set ---")
    preds = model.predict(X_val_scaled)
    print(f"Accuracy: {accuracy_score(y_val, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, preds, target_names=['Home Win', 'Away Win', 'Draw']))
    
    return model, scaler



# --- Main execution block ---
if __name__ == "__main__":
    try:
        # --- Phase 1: Feature Engineering ---
        master_training_df = create_training_data(TRAINING_DIR)
        
        # --- Phase 2: Model Training ---
        trained_model, feature_scaler = train_model(master_training_df)
        
        # --- Phase 3: Save Artifacts ---
        print(f"\nSaving model and scaler to '{MODEL_OUTPUT_DIR}'...")
        MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
        
        joblib.dump(trained_model, MODEL_OUTPUT_DIR / "xgboost_model.joblib")
        joblib.dump(feature_scaler, MODEL_OUTPUT_DIR / "feature_scaler.joblib")
        
        print("\nProcess complete! Model artifacts are ready for the backtest script.")
        
    except FileNotFoundError:
        print(f"\nERROR: Training data directory not found at '{TRAINING_DIR}'.")
        print("Please ensure the path is correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")