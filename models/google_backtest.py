import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import warnings

# Suppress benign warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')


class ProfitabilityBacktester:
    """
    Backtests an LSTM model's betting profitability at specific in-match time points.

    This class loads a pre-trained Keras model and a Scikit-learn scaler,
    iterates through backtest match data, simulates bets at specified times,
    and calculates profitability metrics for each time-based strategy.
    """

    def __init__(self, model_path: str, backtest_dir: str, sequence_length: int, stake: float = 10.0):
        """
        Initializes the backtester.

        Args:
            model_path (str): Path to the directory containing the saved model and scaler.
            backtest_dir (str): Path to the directory with backtest JSON match files.
            sequence_length (int): The sequence length the model was trained on.
            stake (float): The amount to bet on each prediction, in pounds.
        """
        self.model_path = Path(model_path)
        self.backtest_dir = Path(backtest_dir)
        self.sequence_length = sequence_length
        self.features = 4  # home_odds, away_odds, draw_odds, score_diff
        self.stake = stake
        self.betting_results = []

        # Load the trained model and scaler
        print(f"Loading model from: {self.model_path}")
        self.model = load_model(self.model_path / "lstm_model.h5", compile=False)
        self.scaler = joblib.load(self.model_path / "scaler.pkl")
        print("Model and scaler loaded successfully.")

    def _get_team_names(self, match_name: str, odds_keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Robustly determines home and away team names.

        It first tries to parse "Home Team vs Away Team". If that fails, it uses
        a fallback method but issues a warning.

        Args:
            match_name (str): The name of the match (e.g., "Team A vs Team B").
            odds_keys (List[str]): A list of keys from an odds dictionary.

        Returns:
            A tuple containing (home_team_name, away_team_name).
        """
        # Primary method: Parse "vs"
        if ' vs ' in match_name:
            teams = match_name.split(' vs ')
            return teams[0].strip(), teams[1].strip()

        # Fallback method: Use order from odds, which is less reliable
        non_draw_keys = [k for k in odds_keys if k.lower() != 'draw']
        if len(non_draw_keys) >= 2:
            warnings.warn(
                f"Could not parse 'Home vs Away' from match name '{match_name}'. "
                f"Falling back to using the order of teams in the odds data. "
                f"Assuming Home: '{non_draw_keys[0]}', Away: '{non_draw_keys[1]}'. "
                f"PLEASE VALIDATE THIS IS CORRECT."
            )
            return non_draw_keys[0], non_draw_keys[1]

        # If all else fails
        return None, None

    def _extract_match_data(self, match_json: Path) -> Tuple[Optional[pd.DataFrame], Optional[int], str]:
        """
        Loads, cleans, and structures data from a single match JSON file.
        """
        with open(match_json, 'r') as f:
            data = json.load(f)

        if not data:
            return None, None, "Unknown Match"

        match_name = data[0].get('match', 'Unknown Match')

        # Determine final outcome
        final_score_str = data[-1]['score']
        final_home, final_away = map(int, final_score_str.split(' - '))
        if final_home > final_away:
            actual_outcome = 0  # Home win
        elif final_away > final_home:
            actual_outcome = 1  # Away win
        else:
            actual_outcome = 2  # Draw

        # Process each time interval
        processed_data = []
        # Get team names once from the first entry's odds
        home_team, away_team = self._get_team_names(match_name, list(data[0]['odds'].values())[0].keys())

        if not home_team:
            print(f"Skipping match {match_name} due to inability to identify teams.")
            return None, None, match_name

        for i, entry in enumerate(data):
            score_diff = int(entry['score'].split(' - ')[0]) - int(entry['score'].split(' - ')[1])
            
            # Aggregate odds
            h_odds, a_odds, d_odds = [], [], []
            for bookmaker_odds in entry['odds'].values():
                h_odds.append(bookmaker_odds.get(home_team, np.nan))
                a_odds.append(bookmaker_odds.get(away_team, np.nan))
                d_odds.append(bookmaker_odds.get('Draw', np.nan))

            processed_data.append({
                'interval': i,
                'home_odds': np.nanmean(h_odds) if any(~np.isnan(h_odds)) else 2.0,
                'away_odds': np.nanmean(a_odds) if any(~np.isnan(a_odds)) else 2.0,
                'draw_odds': np.nanmean(d_odds) if any(~np.isnan(d_odds)) else 3.0,
                'score_diff': score_diff
            })

        return pd.DataFrame(processed_data), actual_outcome, match_name

    def run_backtest(self, prediction_minutes: List[int]):
        """
        Runs the full backtest across all matches in the backtest directory.

        Args:
            prediction_minutes (List[int]): A list of minutes at which to simulate a bet.
        """
        json_files = list(self.backtest_dir.glob('*.json'))
        print(f"\nStarting backtest on {len(json_files)} matches...")
        print(f"Betting strategies to be tested: {prediction_minutes} minutes")
        print(f"Stake per bet: £{self.stake:.2f}")
        print("-" * 60)

        for i, match_file in enumerate(json_files):
            df, actual_outcome, match_name = self._extract_match_data(match_file)

            if df is None:
                continue

            # Assuming ~40 seconds per interval in the data
            intervals_per_minute = 60 / 40

            for pred_minute in prediction_minutes:
                pred_interval = int(pred_minute * intervals_per_minute)

                # Check if there is enough historical data for a prediction
                if pred_interval < self.sequence_length or pred_interval >= len(df):
                    continue

                # Get the sequence of features leading up to the prediction time
                sequence_df = df.iloc[pred_interval - self.sequence_length + 1 : pred_interval + 1]
                features = sequence_df[['home_odds', 'away_odds', 'draw_odds', 'score_diff']].values

                # Normalize features and predict
                features_normalized = self.scaler.transform(features)
                input_tensor = features_normalized.reshape(1, self.sequence_length, self.features)
                probabilities = self.model.predict(input_tensor, verbose=0)[0]
                predicted_outcome = np.argmax(probabilities)

                # Get odds at the exact time of the bet
                odds_at_bet_time = df.iloc[pred_interval]
                bet_on_map = {0: 'home_odds', 1: 'away_odds', 2: 'draw_odds'}
                bet_odds = odds_at_bet_time[bet_on_map[predicted_outcome]]

                # Calculate Profit and Loss (P&L)
                if predicted_outcome == actual_outcome:
                    pnl = (self.stake * bet_odds) - self.stake
                else:
                    pnl = -self.stake
                
                # Store the result
                self.betting_results.append({
                    'match_name': match_name,
                    'prediction_minute': pred_minute,
                    'predicted_outcome': ['Home', 'Away', 'Draw'][predicted_outcome],
                    'actual_outcome': ['Home', 'Away', 'Draw'][actual_outcome],
                    'bet_odds': bet_odds,
                    'pnl': pnl,
                    'correct_bet': predicted_outcome == actual_outcome
                })
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(json_files)} matches...")
        
        print("-" * 60)
        print("Backtest processing complete.")

    def generate_report(self) -> Optional[pd.DataFrame]:
        """
        Calculates and prints profitability metrics for each independent strategy.
        """
        if not self.betting_results:
            print("No betting results to report. Did the backtest run correctly?")
            return None

        results_df = pd.DataFrame(self.betting_results)

        # --- Use modern .agg() for clear, efficient analysis by strategy ---
        summary_by_time = results_df.groupby('prediction_minute').agg(
            Total_Bets=('pnl', 'size'),
            Total_PL_Pounds=('pnl', 'sum'),
            Avg_Odds=('bet_odds', 'mean'),
            Win_Rate_Percent=('correct_bet', lambda x: x.mean() * 100)
        ).reset_index()
        
        # Calculate Staked Amount and ROI afterwards
        summary_by_time['Total_Staked_Pounds'] = summary_by_time['Total_Bets'] * self.stake
        summary_by_time['ROI_Percent'] = (summary_by_time['Total_PL_Pounds'] / summary_by_time['Total_Staked_Pounds']) * 100

        # --- Print Report ---
        report_header = [
            "="*80,
            "PROFITABILITY BACKTEST REPORT: COMPARING INDEPENDENT STRATEGIES",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*80,
            "\nEach row represents a completely separate betting strategy.",
        ]
        
        # Reorder columns for clarity
        column_order = [
            'prediction_minute', 'Total_Bets', 'Total_Staked_Pounds', 'Total_PL_Pounds',
            'ROI_Percent', 'Win_Rate_Percent', 'Avg_Odds'
        ]
        summary_by_time = summary_by_time[column_order]

        print("\n".join(report_header))
        print(summary_by_time.round(2).to_string(index=False))
        print("="*80)

        return summary_by_time

    def plot_results(self, summary_df: pd.DataFrame, save_path: str = None):
        """
        Creates and shows a plot summarizing the backtest results.
        """
        if summary_df is None or summary_df.empty:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Plot Total P&L as bars
        # MODIFIED: Added hue and legend=False to resolve FutureWarning
        sns.barplot(
            x='prediction_minute', 
            y='Total_PL_Pounds', 
            data=summary_df, 
            ax=ax1, 
            palette="viridis", 
            hue='prediction_minute', 
            legend=False
        )
        ax1.set_xlabel("Betting Strategy (Time in Minutes)", fontsize=12)
        ax1.set_ylabel("Total P&L (£)", fontsize=12)
        ax1.set_title("Performance Comparison of Betting Strategies", fontsize=16, weight='bold')

        # Create a second y-axis for ROI
        ax2 = ax1.twinx()
        sns.lineplot(
            x=ax1.get_xticks(), 
            y='ROI_Percent', 
            data=summary_df, 
            ax=ax2, 
            color='r', 
            marker='o', 
            label='ROI (%)'
        )
        ax2.set_ylabel("Return on Investment (ROI %)", color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"\nPlot saved to: {save_path}")

        plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Choose which trained model to use (e.g., 5, 10, or 20)
    SEQ_LENGTH = 5 
    
    # Define paths and parameters
    MODEL_DIRECTORY = f"./lstm_seq{SEQ_LENGTH}"
    BACKTEST_DATA_DIR = "../data/Backtest" #<-- MAKE SURE THIS PATH IS CORRECT
    STAKE_PER_BET = 10.0
    
    # Define the independent time-based strategies you want to test
    # MODIFIED: Added 10 and 20 minute strategies to the list
    PREDICTION_STRATEGIES = [10, 20, 30, 45, 60]

    # --- RUN BACKTEST ---
    try:
        backtester = ProfitabilityBacktester(
            model_path=MODEL_DIRECTORY,
            backtest_dir=BACKTEST_DATA_DIR,
            sequence_length=SEQ_LENGTH,
            stake=STAKE_PER_BET
        )
        
        backtester.run_backtest(prediction_minutes=PREDICTION_STRATEGIES)
        
        # Generate the report and get the summary data for plotting
        summary_results = backtester.generate_report()
        
        # Plot the results if the backtest was successful
        if summary_results is not None:
            backtester.plot_results(
                summary_results, 
                save_path=f"./profitability_by_time_seq{SEQ_LENGTH}.png"
            )

    except FileNotFoundError:
        print("\nERROR: A directory or file was not found.")
        print(f"Please check that your model directory ('{MODEL_DIRECTORY}')")
        print(f"and your backtest data directory ('{BACKTEST_DATA_DIR}') exist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")