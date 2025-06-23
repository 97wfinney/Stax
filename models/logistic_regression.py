import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
from tqdm import tqdm
import argparse

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. CONFIGURATION ---
TRAINING_DIR = Path("../data/Training")
MODEL_OUTPUT_DIR = Path("./logistic_regression_model")

def get_team_names(match_name: str, odds_data: dict) -> Tuple[Optional[str], Optional[str]]:
    """Robustly determines home and away team names from match data."""
    if ' vs ' in match_name:
        teams = match_name.split(' vs ')
        home_team, away_team = teams[0].strip(), teams[1].strip()
        if home_team and away_team:
            return home_team, away_team
    all_potential_teams = {k.strip() for d in odds_data.values() for k in d if isinstance(k, str) and k.strip().lower() != 'draw'}
    sorted_teams = sorted(list(all_potential_teams))
    if len(sorted_teams) >= 2:
        return sorted_teams[0], sorted_teams[1]
    return None, None

def engineer_features_for_lr(data: List[Dict]) -> Optional[pd.DataFrame]:
    """
    Engineers the specific, simple feature set for the Logistic Regression model.
    """
    if not data: return None
    match_name = data[0].get('match', 'Unknown Match')
    home_team, away_team = get_team_names(match_name, data[0].get('odds', {}))
    if not home_team or not away_team:
        print(f"\nWarning: Could not determine teams for '{match_name}'. Skipping file.")
        return None

    final_home, final_away = map(int, data[-1]['score'].split(' - '))
    if final_home > final_away: final_outcome = 0
    elif final_away > final_home: final_outcome = 1
    else: final_outcome = 2

    rows = []
    for entry in data:
        home_score, away_score = map(int, entry['score'].split(' - '))
        h_odds, a_odds, d_odds = [], [], []
        for bookmaker_odds in entry.get('odds', {}).values():
            h_odds.append(bookmaker_odds.get(home_team))
            a_odds.append(bookmaker_odds.get(away_team))
            d_odds.append(bookmaker_odds.get('Draw'))
        
        h_odds = [o for o in h_odds if o is not None]
        a_odds = [o for o in a_odds if o is not None]
        d_odds = [o for o in d_odds if o is not None]

        rows.append({
            'home_score': home_score,
            'away_score': away_score,
            'avg_home_odds': np.mean(h_odds) if h_odds else np.nan,
            'avg_away_odds': np.mean(a_odds) if a_odds else np.nan,
            'avg_draw_odds': np.mean(d_odds) if d_odds else np.nan,
        })
    df = pd.DataFrame(rows)
    df['final_outcome'] = final_outcome
    return df

def run_training_pipeline():
    """
    Full pipeline to process data, find best hyperparameters, train, and save the model.
    """
    print("--- Starting Logistic Regression Training Pipeline ---")
    
    # 1. Create Master Training Dataset
    json_files = list(TRAINING_DIR.glob('*.json'))
    print(f"Found {len(json_files)} match files in {TRAINING_DIR}.")
    
    all_dfs = [df for file_path in tqdm(json_files, desc="Processing training files") 
               if (df := engineer_features_for_lr(json.load(open(file_path, 'r')))) is not None]
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.dropna(inplace=True)
    print(f"\nMaster dataset created with {len(master_df)} samples.")
    
    # 2. Define Features and Target
    features = ['home_score', 'away_score', 'avg_home_odds', 'avg_away_odds', 'avg_draw_odds']
    X = master_df[features]
    y = master_df['final_outcome']
    
    # 3. Scale Features (Essential for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Hyperparameter Tuning with GridSearchCV
    print("\nFinding best hyperparameters using GridSearchCV...")
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    best_c = grid_search.best_params_['C']
    print(f"Best 'C' parameter found: {best_c}")
    
    # 5. Train Final Model with Best Parameter
    print("\nTraining final model on the full dataset with best parameters...")
    final_model = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=1000, C=best_c, random_state=42
    )
    final_model.fit(X_scaled, y)
    
    # Optional: Evaluate on a hold-out set for final confirmation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    preds = final_model.predict(X_test)
    print("\n--- Final Model Evaluation on a Hold-Out Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=['Home Win', 'Away Win', 'Draw']))

    # 6. Save Model and Scaler
    print(f"\nSaving model and scaler to '{MODEL_OUTPUT_DIR}'...")
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_OUTPUT_DIR / "logistic_regression_model.joblib")
    joblib.dump(scaler, MODEL_OUTPUT_DIR / "feature_scaler.joblib")
    
    print("\n--- Training pipeline complete! ---")

if __name__ == "__main__":
    run_training_pipeline()
