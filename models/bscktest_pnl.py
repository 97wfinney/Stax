import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime


class ProfitabilityBacktester:
    """
    Backtest LSTM model focusing on betting profitability using actual odds.
    """
    
    def __init__(self, model_path: str = "./lstm_seq5", 
                 sequence_length: int = 5,
                 backtest_dir: str = "../data/Backtest",
                 stake: float = 10.0):
        """
        Initialize backtester with the seq5 model.
        
        Args:
            model_path: Path to saved model directory (default: lstm_seq5)
            sequence_length: Sequence length (default: 5)
            backtest_dir: Path to backtest data directory
            stake: Bet amount per match in £ (default: £10)
        """
        self.sequence_length = sequence_length
        self.backtest_dir = Path(backtest_dir)
        self.features = 4
        self.stake = stake
        
        # Load model and scaler
        print(f"Loading model from {model_path}...")
        self.model = load_model(f"{model_path}/lstm_model.h5")
        self.scaler = joblib.load(f"{model_path}/scaler.pkl")
        
        # Results storage
        self.betting_results = []
        self.cumulative_pnl = []
        
    def extract_features_and_odds(self, match_data: List[Dict], interval_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Extract features and actual odds at a specific interval.
        """
        features = []
        
        # Get features up to the specified interval
        for i in range(min(interval_idx + 1, len(match_data))):
            entry = match_data[i]
            
            # Extract score
            score_parts = entry['score'].split(' - ')
            home_score = int(score_parts[0])
            away_score = int(score_parts[1])
            score_diff = home_score - away_score
            
            # Calculate average odds
            odds_dict = entry['odds']
            home_odds = []
            away_odds = []
            draw_odds = []
            
            for bookmaker, odds in odds_dict.items():
                odds_keys = list(odds.keys())
                team_keys = [k for k in odds_keys if k != 'Draw']
                
                if len(team_keys) >= 2:
                    home_odds.append(odds[team_keys[0]])
                    away_odds.append(odds[team_keys[1]])
                elif len(team_keys) == 1:
                    home_odds.append(odds[team_keys[0]])
                    
                if 'Draw' in odds:
                    draw_odds.append(odds['Draw'])
            
            avg_home_odds = np.mean(home_odds) if home_odds else 2.0
            avg_away_odds = np.mean(away_odds) if away_odds else 2.0
            avg_draw_odds = np.mean(draw_odds) if draw_odds else 3.0
            
            features.append([avg_home_odds, avg_away_odds, avg_draw_odds, score_diff])
        
        # Get the actual odds at betting time (average across bookmakers)
        current_odds = match_data[interval_idx]['odds']
        
        # Calculate average odds properly
        home_odds_list = []
        away_odds_list = []
        draw_odds_list = []
        
        # We need to identify home/away teams from the match name
        match_name = match_data[0]['match']
        # Extract team names - typically "HomeTeam vs AwayTeam"
        if ' vs ' in match_name:
            teams = match_name.split(' vs ')
            home_team = teams[0].strip()
            away_team = teams[1].strip()
        else:
            # Fallback - use first two non-Draw keys
            all_team_names = set()
            for odds in current_odds.values():
                all_team_names.update([k for k in odds.keys() if k != 'Draw'])
            team_list = sorted(list(all_team_names))
            home_team = team_list[0] if len(team_list) > 0 else ""
            away_team = team_list[1] if len(team_list) > 1 else ""
        
        # Now extract odds properly
        for bookmaker, odds in current_odds.items():
            if home_team in odds:
                home_odds_list.append(odds[home_team])
            if away_team in odds:
                away_odds_list.append(odds[away_team])
            if 'Draw' in odds:
                draw_odds_list.append(odds['Draw'])
        
        # Use average odds (more realistic than best odds)
        best_odds = {
            'home': np.mean(home_odds_list) if home_odds_list else 2.0,
            'away': np.mean(away_odds_list) if away_odds_list else 3.0,
            'draw': np.mean(draw_odds_list) if draw_odds_list else 3.5
        }
        
        # Sanity check - odds should be reasonable (between 1.1 and 20)
        for outcome in ['home', 'away', 'draw']:
            if best_odds[outcome] < 1.1:
                best_odds[outcome] = 1.1
            elif best_odds[outcome] > 20:
                best_odds[outcome] = 20
        
        return np.array(features), best_odds
    
    def evaluate_match_profitability(self, json_path: str, prediction_minutes: List[int]) -> List[Dict]:
        """
        Evaluate betting profitability at different time points in the match.
        """
        with open(json_path, 'r') as f:
            match_data = json.load(f)
        
        # Get final outcome
        final_score = match_data[-1]['score'].split(' - ')
        final_home = int(final_score[0])
        final_away = int(final_score[1])
        
        if final_home > final_away:
            actual_outcome = 0  # Home win
        elif final_away > final_home:
            actual_outcome = 1  # Away win
        else:
            actual_outcome = 2  # Draw
        
        match_results = []
        
        for pred_minute in prediction_minutes:
            pred_interval = int(pred_minute * 60 / 40)
            
            # Skip if not enough data
            if pred_interval < self.sequence_length or pred_interval >= len(match_data):
                continue
            
            # Extract features and odds
            features, best_odds = self.extract_features_and_odds(match_data, pred_interval)
            
            # Make prediction
            sequence = features[pred_interval-self.sequence_length+1:pred_interval+1]
            sequence_normalized = self.scaler.transform(sequence)
            sequence_normalized = sequence_normalized.reshape(1, self.sequence_length, self.features)
            
            probabilities = self.model.predict(sequence_normalized, verbose=0)[0]
            predicted_outcome = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            # Calculate profit/loss
            if predicted_outcome == 0:  # Bet on home
                bet_odds = best_odds['home']
            elif predicted_outcome == 1:  # Bet on away
                bet_odds = best_odds['away']
            else:  # Bet on draw
                bet_odds = best_odds['draw']
            
            # Calculate P&L
            if predicted_outcome == actual_outcome:
                # Win: (stake * odds) - stake
                pnl = (self.stake * bet_odds) - self.stake
            else:
                # Loss: -stake
                pnl = -self.stake
            
            result = {
                'match': match_data[0]['match'],
                'prediction_minute': pred_minute,
                'score_at_bet': match_data[pred_interval]['score'],
                'final_score': match_data[-1]['score'],
                'predicted_outcome': predicted_outcome,
                'actual_outcome': actual_outcome,
                'probabilities': probabilities.tolist(),
                'confidence': confidence,
                'bet_odds': bet_odds,
                'stake': self.stake,
                'pnl': pnl,
                'correct': predicted_outcome == actual_outcome
            }
            
            match_results.append(result)
            self.betting_results.append(result)
        
        return match_results
    
    def run_profitability_backtest(self, prediction_minutes: List[int] = [30, 45, 60]) -> Dict:
        """
        Run comprehensive profitability backtest.
        """
        json_files = list(self.backtest_dir.glob('*.json'))
        print(f"Running profitability backtest on {len(json_files)} matches...")
        print(f"Stake per bet: £{self.stake}")
        print(f"Prediction times: {prediction_minutes} minutes")
        print("-" * 60)
        
        for i, json_file in enumerate(json_files):
            try:
                self.evaluate_match_profitability(str(json_file), prediction_minutes)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(json_files)} matches")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # Calculate summary statistics
        return self.calculate_profitability_metrics()
    
    def calculate_profitability_metrics(self) -> Dict:
        """
        Calculate comprehensive profitability metrics.
        """
        df = pd.DataFrame(self.betting_results)
        
        # Overall metrics
        total_bets = len(df)
        total_stake = df['stake'].sum()
        total_pnl = df['pnl'].sum()
        winning_bets = df['correct'].sum()
        
        # Metrics by prediction time
        time_metrics = {}
        for minute in df['prediction_minute'].unique():
            minute_df = df[df['prediction_minute'] == minute]
            time_metrics[f'{minute}_min'] = {
                'bets': len(minute_df),
                'wins': minute_df['correct'].sum(),
                'win_rate': minute_df['correct'].mean(),
                'total_pnl': minute_df['pnl'].sum(),
                'roi': minute_df['pnl'].sum() / minute_df['stake'].sum(),
                'avg_odds': minute_df['bet_odds'].mean(),
                'avg_confidence': minute_df['confidence'].mean()
            }
        
        # Metrics by outcome type
        outcome_metrics = {}
        for outcome in [0, 1, 2]:
            outcome_df = df[df['predicted_outcome'] == outcome]
            outcome_name = ['home', 'away', 'draw'][outcome]
            outcome_metrics[outcome_name] = {
                'bets': len(outcome_df),
                'wins': outcome_df['correct'].sum(),
                'win_rate': outcome_df['correct'].mean() if len(outcome_df) > 0 else 0,
                'total_pnl': outcome_df['pnl'].sum() if len(outcome_df) > 0 else 0,
                'roi': outcome_df['pnl'].sum() / outcome_df['stake'].sum() if len(outcome_df) > 0 else 0,
                'avg_odds': outcome_df['bet_odds'].mean() if len(outcome_df) > 0 else 0
            }
        
        # Calculate cumulative P&L
        df_sorted = df.sort_index()
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        self.cumulative_pnl = df_sorted['cumulative_pnl'].tolist()
        
        # High confidence bets (>0.7)
        high_conf_df = df[df['confidence'] > 0.7]
        
        results = {
            'total_bets': total_bets,
            'total_stake': total_stake,
            'total_pnl': total_pnl,
            'roi': total_pnl / total_stake,
            'winning_bets': winning_bets,
            'win_rate': winning_bets / total_bets,
            'avg_odds': df['bet_odds'].mean(),
            'time_metrics': time_metrics,
            'outcome_metrics': outcome_metrics,
            'high_confidence': {
                'bets': len(high_conf_df),
                'wins': high_conf_df['correct'].sum(),
                'win_rate': high_conf_df['correct'].mean() if len(high_conf_df) > 0 else 0,
                'pnl': high_conf_df['pnl'].sum() if len(high_conf_df) > 0 else 0,
                'roi': high_conf_df['pnl'].sum() / high_conf_df['stake'].sum() if len(high_conf_df) > 0 else 0
            },
            'max_drawdown': self.calculate_max_drawdown(),
            'sharpe_ratio': self.calculate_sharpe_ratio(df)
        }
        
        return results
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from cumulative P&L."""
        if not self.cumulative_pnl:
            return 0
        
        cumsum = np.array(self.cumulative_pnl)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = cumsum - running_max
        return float(np.min(drawdown))
    
    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio (simplified - assumes 0 risk-free rate)."""
        if len(df) < 2:
            return 0
        
        returns = df['pnl'] / df['stake']
        return float(returns.mean() / returns.std() if returns.std() > 0 else 0)
    
    def generate_profitability_report(self, results: Dict, output_path: str = None):
        """Generate detailed profitability report."""
        report = []
        report.append("="*60)
        report.append("LSTM MODEL PROFITABILITY BACKTEST REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: lstm_seq5 | Stake: £{self.stake} per bet")
        report.append("")
        
        # Overall Performance
        report.append("OVERALL PERFORMANCE")
        report.append("-"*30)
        report.append(f"Total Bets: {results['total_bets']}")
        report.append(f"Winning Bets: {results['winning_bets']} ({results['win_rate']:.1%})")
        report.append(f"Total Stake: £{results['total_stake']:,.2f}")
        report.append(f"Total P&L: £{results['total_pnl']:,.2f}")
        report.append(f"ROI: {results['roi']:.1%}")
        report.append(f"Average Odds: {results['avg_odds']:.2f}")
        report.append(f"Max Drawdown: £{results['max_drawdown']:,.2f}")
        report.append(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        report.append("")
        
        # Performance by Prediction Time
        report.append("PERFORMANCE BY PREDICTION TIME")
        report.append("-"*30)
        for time, metrics in results['time_metrics'].items():
            report.append(f"\n{time}:")
            report.append(f"  Bets: {metrics['bets']}")
            report.append(f"  Win Rate: {metrics['win_rate']:.1%}")
            report.append(f"  P&L: £{metrics['total_pnl']:,.2f}")
            report.append(f"  ROI: {metrics['roi']:.1%}")
            report.append(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
        
        # Performance by Outcome Type
        report.append("\nPERFORMANCE BY BET TYPE")
        report.append("-"*30)
        for outcome, metrics in results['outcome_metrics'].items():
            report.append(f"\n{outcome.upper()}:")
            report.append(f"  Bets: {metrics['bets']}")
            report.append(f"  Win Rate: {metrics['win_rate']:.1%}")
            report.append(f"  P&L: £{metrics['total_pnl']:,.2f}")
            report.append(f"  ROI: {metrics['roi']:.1%}")
            report.append(f"  Avg Odds: {metrics['avg_odds']:.2f}")
        
        # High Confidence Bets
        report.append("\nHIGH CONFIDENCE BETS (>70%)")
        report.append("-"*30)
        hc = results['high_confidence']
        report.append(f"Bets: {hc['bets']}")
        report.append(f"Win Rate: {hc['win_rate']:.1%}")
        report.append(f"P&L: £{hc['pnl']:,.2f}")
        report.append(f"ROI: {hc['roi']:.1%}")
        
        report_text = "\n".join(report)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text
    
    def plot_profitability(self, results: Dict, save_path: str = None):
        """Create profitability visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cumulative P&L over time
        axes[0, 0].plot(self.cumulative_pnl, linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].fill_between(range(len(self.cumulative_pnl)), 0, self.cumulative_pnl, 
                               where=[x > 0 for x in self.cumulative_pnl], 
                               color='green', alpha=0.3, label='Profit')
        axes[0, 0].fill_between(range(len(self.cumulative_pnl)), 0, self.cumulative_pnl, 
                               where=[x <= 0 for x in self.cumulative_pnl], 
                               color='red', alpha=0.3, label='Loss')
        axes[0, 0].set_xlabel('Bet Number')
        axes[0, 0].set_ylabel('Cumulative P&L (£)')
        axes[0, 0].set_title('Cumulative Profit/Loss Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. P&L by prediction time
        times = []
        pnls = []
        win_rates = []
        for time, metrics in results['time_metrics'].items():
            times.append(time)
            pnls.append(metrics['total_pnl'])
            win_rates.append(metrics['win_rate'] * 100)
        
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        
        bars = ax2.bar(times, pnls, alpha=0.7, color=['green' if p > 0 else 'red' for p in pnls])
        line = ax2_twin.plot(times, win_rates, 'b-o', linewidth=2, markersize=8, label='Win Rate')
        
        ax2.set_xlabel('Prediction Time')
        ax2.set_ylabel('Total P&L (£)', color='black')
        ax2_twin.set_ylabel('Win Rate (%)', color='blue')
        ax2.set_title('P&L and Win Rate by Prediction Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. P&L by bet type
        outcomes = list(results['outcome_metrics'].keys())
        outcome_pnls = [results['outcome_metrics'][o]['total_pnl'] for o in outcomes]
        outcome_bets = [results['outcome_metrics'][o]['bets'] for o in outcomes]
        
        axes[1, 0].bar(outcomes, outcome_pnls, color=['green' if p > 0 else 'red' for p in outcome_pnls])
        axes[1, 0].set_xlabel('Bet Type')
        axes[1, 0].set_ylabel('Total P&L (£)')
        axes[1, 0].set_title('Profitability by Bet Type')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add bet counts on bars
        for i, (pnl, bets) in enumerate(zip(outcome_pnls, outcome_bets)):
            axes[1, 0].text(i, pnl + (50 if pnl > 0 else -50), f'{bets} bets', 
                           ha='center', va='bottom' if pnl > 0 else 'top')
        
        # 4. Win rate vs confidence
        df = pd.DataFrame(self.betting_results)
        confidence_bins = pd.cut(df['confidence'], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        conf_win_rate = df.groupby(confidence_bins)['correct'].agg(['mean', 'count'])
        
        axes[1, 1].bar(range(len(conf_win_rate)), conf_win_rate['mean'] * 100, 
                      tick_label=[str(i) for i in conf_win_rate.index])
        axes[1, 1].set_xlabel('Confidence Level')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].set_title('Win Rate by Confidence Level')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for i, (rate, count) in enumerate(zip(conf_win_rate['mean'], conf_win_rate['count'])):
            axes[1, 1].text(i, rate * 100 + 1, f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Run the profitability backtest
if __name__ == "__main__":
    # Initialize backtester with lstm_seq5 model
    backtester = ProfitabilityBacktester(
        model_path="./lstm_seq5",
        sequence_length=5,
        backtest_dir="../data/Backtest",
        stake=10.0  # £10 per bet
    )
    
    # Run backtest at different time points
    results = backtester.run_profitability_backtest(
        prediction_minutes=[30, 45, 60]  # Test at 30 mins, halftime, and 60 mins
    )
    
    # Generate report
    backtester.generate_profitability_report(
        results, 
        output_path="./profitability_report_lstm_seq5.txt"
    )
    
    # Generate visualization
    backtester.plot_profitability(
        results,
        save_path="./profitability_plots_lstm_seq5.png"
    )
    
    print("\n" + "="*60)
    print("Backtest complete!")
    print(f"Total P&L: £{results['total_pnl']:,.2f}")
    print(f"ROI: {results['roi']:.1%}")
    print("Report saved to: profitability_report_lstm_seq5.txt")
    print("Plots saved to: profitability_plots_lstm_seq5.png")