#!/usr/bin/env python3
"""
Advanced Backtest Analytics - Deep dive analysis of model performance
across different match states, odds ranges, and parameter combinations.

This script analyzes existing backtest results to find optimal betting conditions
and parameter settings.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
from typing import Dict, List, Tuple
from datetime import datetime
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STAX_MODEL_DIR = MODELS_DIR / "stax_kfold"
ANALYTICS_DIR = Path(__file__).resolve().parent / "analytics_results"
ANALYTICS_DIR.mkdir(exist_ok=True)

# Import necessary components
from acca import StaxModelLoader
from confidence_trigger_backtest import ConfidenceTriggerBacktester


class AdvancedBacktestAnalytics:
    """Comprehensive analytics for backtest results."""
    
    def __init__(self):
        self.stax_model = StaxModelLoader()
        self.stax_model.load_saved_models()
        self.all_results = []
        self.analytics_results = {}
        
    def run_comprehensive_backtest(self, confidence_thresholds: List[float], 
                                  kelly_fractions: List[float], 
                                  initial_bankroll: float = 1000.0,
                                  max_bet_fractions: List[float] = [0.05]):
        """Run backtests across multiple parameter combinations."""
        print("Running comprehensive backtests...")
        backtest_dir = DATA_DIR / "Backtest"
        
        total_combinations = len(confidence_thresholds) * len(kelly_fractions) * len(max_bet_fractions)
        current = 0
        
        for confidence, kelly, max_bet in product(confidence_thresholds, kelly_fractions, max_bet_fractions):
            current += 1
            print(f"\nTesting combination {current}/{total_combinations}")
            print(f"  Confidence: {confidence}%, Kelly: {kelly}, Max Bet: {max_bet*100}%")
            
            # Run backtest with current parameters
            backtester = ConfidenceTriggerBacktester(
                self.stax_model,
                confidence,
                kelly_fraction=kelly if kelly > 0 else None,
                initial_bankroll=initial_bankroll,
                max_bet_fraction=max_bet
            )
            
            # Process all matches
            match_files = list(backtest_dir.glob('*.json'))
            results_with_params = []
            
            for match_file in match_files:
                with open(match_file, 'r') as f:
                    match_data = json.load(f)
                
                # Get match result with full details
                result = self.process_match_with_details(
                    backtester, match_file, match_data, 
                    confidence, kelly, max_bet
                )
                results_with_params.append(result)
            
            self.all_results.extend(results_with_params)
        
        # Convert to DataFrame for analysis
        self.df_results = pd.DataFrame(self.all_results)
        print(f"\nCompleted {len(self.all_results)} total simulations")
        
    def process_match_with_details(self, backtester, match_file, match_data, 
                                  confidence, kelly, max_bet):
        """Process a match and extract detailed information."""
        # Get basic prediction
        result = backtester.process_match(match_file)
        
        # Add parameter info
        result['confidence_threshold'] = confidence
        result['kelly_fraction'] = kelly
        result['max_bet_fraction'] = max_bet
        
        # If a bet was placed, get additional match state info
        if result['stake'] > 0:
            # Get match state at betting time
            bet_time = result['bet_time']
            
            # Find the data point closest to bet time
            time_index = int(bet_time * 60 / 40)  # Convert minutes to index
            if time_index < len(match_data):
                match_state = match_data[time_index]
                scores = match_state['score'].split(' - ')
                home_score = int(scores[0])
                away_score = int(scores[1])
                
                result['score_diff_at_bet'] = home_score - away_score
                result['total_goals_at_bet'] = home_score + away_score
                result['home_score_at_bet'] = home_score
                result['away_score_at_bet'] = away_score
                
                # Categorize match state
                if home_score > away_score:
                    result['match_state'] = 'home_leading'
                elif away_score > home_score:
                    result['match_state'] = 'away_leading'
                else:
                    result['match_state'] = 'draw'
            else:
                result['score_diff_at_bet'] = 0
                result['total_goals_at_bet'] = 0
                result['match_state'] = 'unknown'
        else:
            result['score_diff_at_bet'] = None
            result['total_goals_at_bet'] = None
            result['match_state'] = None
            
        return result
    
    def analyze_score_differential(self):
        """Analyze P&L by score differential at time of bet."""
        print("\n=== Score Differential Analysis ===")
        
        # Filter to only bets placed
        df_bets = self.df_results[self.df_results['stake'] > 0].copy()
        
        if len(df_bets) == 0:
            print("No bets placed in dataset")
            return
        
        # Group by score differential
        score_diff_analysis = df_bets.groupby('score_diff_at_bet').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_correct': 'mean',
            'odds': 'mean',
            'confidence': 'mean'
        }).round(2)
        
        score_diff_analysis.columns = ['num_bets', 'total_pnl', 'avg_pnl', 
                                       'win_rate', 'avg_odds', 'avg_confidence']
        
        print("\nP&L by Score Differential:")
        print(score_diff_analysis)
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total P&L by score diff
        score_diff_analysis['total_pnl'].plot(kind='bar', ax=axes[0,0], 
                                              color=['red' if x < 0 else 'green' 
                                                     for x in score_diff_analysis['total_pnl']])
        axes[0,0].set_title('Total P&L by Score Differential')
        axes[0,0].set_xlabel('Score Differential (Home - Away)')
        axes[0,0].set_ylabel('Total P&L (£)')
        
        # Win rate by score diff
        score_diff_analysis['win_rate'].plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Win Rate by Score Differential')
        axes[0,1].set_xlabel('Score Differential')
        axes[0,1].set_ylabel('Win Rate')
        axes[0,1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        # Number of bets by score diff
        score_diff_analysis['num_bets'].plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Number of Bets by Score Differential')
        axes[1,0].set_xlabel('Score Differential')
        axes[1,0].set_ylabel('Number of Bets')
        
        # Match state analysis
        match_state_analysis = df_bets.groupby('match_state').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_correct': 'mean'
        })
        match_state_analysis.columns = ['num_bets', 'total_pnl', 'avg_pnl', 'win_rate']
        
        match_state_analysis['total_pnl'].plot(kind='bar', ax=axes[1,1],
                                               color=['red' if x < 0 else 'green' 
                                                      for x in match_state_analysis['total_pnl']])
        axes[1,1].set_title('Total P&L by Match State')
        axes[1,1].set_xlabel('Match State')
        axes[1,1].set_ylabel('Total P&L (£)')
        
        plt.tight_layout()
        plt.savefig(ANALYTICS_DIR / 'score_differential_analysis.png', dpi=300)
        plt.show()
        
        self.analytics_results['score_differential'] = score_diff_analysis
        
    def analyze_odds_ranges(self):
        """Analyze performance across different odds ranges."""
        print("\n=== Odds Range Analysis ===")
        
        df_bets = self.df_results[self.df_results['stake'] > 0].copy()
        
        # Create odds buckets
        odds_bins = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 10.0]
        df_bets['odds_range'] = pd.cut(df_bets['odds'], bins=odds_bins)
        
        # Analyze by odds range
        odds_analysis = df_bets.groupby('odds_range').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_correct': 'mean',
            'confidence': 'mean',
            'stake': 'mean'
        }).round(2)
        
        odds_analysis.columns = ['num_bets', 'total_pnl', 'avg_pnl', 
                                'win_rate', 'avg_confidence', 'avg_stake']
        
        # Calculate ROI
        odds_analysis['roi'] = (odds_analysis['total_pnl'] / 
                                (odds_analysis['num_bets'] * odds_analysis['avg_stake'])) * 100
        
        print("\nPerformance by Odds Range:")
        print(odds_analysis)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ROI by odds range
        odds_analysis['roi'].plot(kind='bar', ax=axes[0,0],
                                  color=['red' if x < 0 else 'green' 
                                         for x in odds_analysis['roi']])
        axes[0,0].set_title('ROI by Odds Range')
        axes[0,0].set_xlabel('Odds Range')
        axes[0,0].set_ylabel('ROI (%)')
        axes[0,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Win rate vs expected
        ax = axes[0,1]
        x_pos = range(len(odds_analysis))
        
        # Calculate expected win rate from odds (middle of range)
        expected_win_rates = []
        for idx in odds_analysis.index:
            mid_odds = (idx.left + idx.right) / 2
            expected_win_rates.append(1 / mid_odds)
        
        ax.bar(x_pos, odds_analysis['win_rate'], alpha=0.7, label='Actual Win Rate')
        ax.plot(x_pos, expected_win_rates, 'r--', marker='o', label='Expected from Odds')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(x) for x in odds_analysis.index], rotation=45)
        ax.set_title('Actual vs Expected Win Rate by Odds Range')
        ax.set_xlabel('Odds Range')
        ax.set_ylabel('Win Rate')
        ax.legend()
        
        # Number of bets distribution
        odds_analysis['num_bets'].plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Bet Distribution by Odds Range')
        axes[1,0].set_xlabel('Odds Range')
        axes[1,0].set_ylabel('Number of Bets')
        
        # Confidence vs Odds scatter
        axes[1,1].scatter(df_bets['odds'], df_bets['confidence'], 
                         c=df_bets['is_correct'], cmap='RdYlGn', alpha=0.5)
        axes[1,1].set_xlabel('Odds')
        axes[1,1].set_ylabel('Model Confidence (%)')
        axes[1,1].set_title('Confidence vs Odds (Green=Win, Red=Loss)')
        
        plt.tight_layout()
        plt.savefig(ANALYTICS_DIR / 'odds_range_analysis.png', dpi=300)
        plt.show()
        
        self.analytics_results['odds_ranges'] = odds_analysis
        
    def analyze_calibration(self):
        """Analyze model confidence calibration."""
        print("\n=== Calibration Analysis ===")
        
        df_bets = self.df_results[self.df_results['stake'] > 0].copy()
        
        # Create confidence bins
        conf_bins = list(range(0, 101, 5))
        df_bets['conf_bin'] = pd.cut(df_bets['confidence'], bins=conf_bins)
        
        # Calculate actual win rate per confidence bin
        calibration_data = df_bets.groupby('conf_bin').agg({
            'is_correct': ['mean', 'count'],
            'pnl': 'sum'
        }).round(3)
        
        calibration_data.columns = ['actual_win_rate', 'num_bets', 'total_pnl']
        
        # Get bin centers for expected win rate
        calibration_data['expected_win_rate'] = [
            (interval.left + interval.right) / 200  # Convert to probability
            for interval in calibration_data.index
        ]
        
        # Calculate calibration error
        calibration_data['calibration_error'] = abs(
            calibration_data['actual_win_rate'] - calibration_data['expected_win_rate']
        )
        
        print("\nCalibration Analysis:")
        print(calibration_data)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration plot
        ax = axes[0]
        x = calibration_data['expected_win_rate'] * 100
        y = calibration_data['actual_win_rate'] * 100
        sizes = calibration_data['num_bets'] * 10  # Scale for visibility
        
        scatter = ax.scatter(x, y, s=sizes, alpha=0.6, 
                           c=calibration_data['total_pnl'], cmap='RdYlGn')
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Model Confidence (%)')
        ax.set_ylabel('Actual Win Rate (%)')
        ax.set_title('Model Calibration (size = num bets, color = P&L)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Total P&L (£)')
        
        # Value identification
        df_bets['expected_value'] = df_bets['confidence']/100 * df_bets['odds'] - 1
        
        # Scatter plot of EV vs actual returns
        ax2 = axes[1]
        returns = df_bets['pnl'] / df_bets['stake']
        ax2.scatter(df_bets['expected_value'], returns, alpha=0.5,
                   c=df_bets['is_correct'], cmap='RdYlGn')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Expected Value')
        ax2.set_ylabel('Actual Return')
        ax2.set_title('Expected Value vs Actual Returns')
        
        plt.tight_layout()
        plt.savefig(ANALYTICS_DIR / 'calibration_analysis.png', dpi=300)
        plt.show()
        
        self.analytics_results['calibration'] = calibration_data
        
    def grid_search_optimization(self):
        """Perform grid search to find optimal parameters."""
        print("\n=== Grid Search Optimization ===")
        
        # Group by parameter combinations
        param_groups = self.df_results.groupby([
            'confidence_threshold', 'kelly_fraction', 'max_bet_fraction'
        ])
        
        optimization_results = []
        
        for params, group in param_groups:
            confidence, kelly, max_bet = params
            
            # Calculate metrics for this combination
            bets = group[group['stake'] > 0]
            
            if len(bets) == 0:
                continue
                
            total_stake = bets['stake'].sum()
            total_pnl = bets['pnl'].sum()
            num_bets = len(bets)
            win_rate = bets['is_correct'].mean()
            
            # Calculate Sharpe ratio (assuming daily returns)
            if len(bets) > 1:
                returns = bets['pnl'] / bets['stake']
                sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe = 0
            
            # Calculate max drawdown
            cumulative_pnl = bets['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max).min()
            
            optimization_results.append({
                'confidence': confidence,
                'kelly': kelly,
                'max_bet': max_bet,
                'num_bets': num_bets,
                'total_pnl': total_pnl,
                'roi': (total_pnl / total_stake * 100) if total_stake > 0 else 0,
                'win_rate': win_rate * 100,
                'sharpe': sharpe,
                'max_drawdown': drawdown,
                'avg_stake': bets['stake'].mean() if len(bets) > 0 else 0
            })
        
        opt_df = pd.DataFrame(optimization_results)
        
        # Sort by different metrics
        print("\nTop 10 by ROI:")
        print(opt_df.nlargest(10, 'roi')[['confidence', 'kelly', 'roi', 'num_bets', 'win_rate']].round(2))
        
        print("\nTop 10 by Sharpe Ratio:")
        print(opt_df.nlargest(10, 'sharpe')[['confidence', 'kelly', 'sharpe', 'roi', 'num_bets']].round(2))
        
        print("\nTop 10 by Total P&L:")
        print(opt_df.nlargest(10, 'total_pnl')[['confidence', 'kelly', 'total_pnl', 'roi', 'num_bets']].round(2))
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Heatmap of ROI
        if len(opt_df['kelly'].unique()) > 1 and len(opt_df['confidence'].unique()) > 1:
            pivot_roi = opt_df.pivot_table(
                values='roi', 
                index='confidence', 
                columns='kelly',
                aggfunc='mean'
            )
            sns.heatmap(pivot_roi, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, ax=axes[0,0])
            axes[0,0].set_title('ROI Heatmap (Confidence vs Kelly)')
        
        # Sharpe vs ROI scatter
        scatter = axes[0,1].scatter(opt_df['roi'], opt_df['sharpe'], 
                                   s=opt_df['num_bets']*2, alpha=0.6,
                                   c=opt_df['confidence'], cmap='viridis')
        axes[0,1].set_xlabel('ROI (%)')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].set_title('Sharpe vs ROI (size = num bets, color = confidence)')
        plt.colorbar(scatter, ax=axes[0,1], label='Confidence %')
        
        # Number of bets vs confidence
        bet_counts = opt_df.groupby('confidence')['num_bets'].mean()
        bet_counts.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Average Number of Bets by Confidence Threshold')
        axes[1,0].set_xlabel('Confidence Threshold (%)')
        axes[1,0].set_ylabel('Average Number of Bets')
        
        # Win rate vs confidence
        win_rates = opt_df.groupby('confidence')['win_rate'].mean()
        win_rates.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Average Win Rate by Confidence Threshold')
        axes[1,1].set_xlabel('Confidence Threshold (%)')
        axes[1,1].set_ylabel('Win Rate (%)')
        axes[1,1].axhline(y=50, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(ANALYTICS_DIR / 'grid_search_optimization.png', dpi=300)
        plt.show()
        
        self.analytics_results['optimization'] = opt_df
        
        # Find best parameter combination
        best_roi = opt_df.loc[opt_df['roi'].idxmax()]
        best_sharpe = opt_df.loc[opt_df['sharpe'].idxmax()]
        
        print("\n=== OPTIMAL PARAMETERS ===")
        print(f"\nBest by ROI:")
        print(f"  Confidence: {best_roi['confidence']}%")
        print(f"  Kelly: {best_roi['kelly']}")
        print(f"  ROI: {best_roi['roi']:.2f}%")
        print(f"  Bets: {best_roi['num_bets']}")
        
        print(f"\nBest by Sharpe Ratio:")
        print(f"  Confidence: {best_sharpe['confidence']}%")
        print(f"  Kelly: {best_sharpe['kelly']}")
        print(f"  Sharpe: {best_sharpe['sharpe']:.2f}")
        print(f"  ROI: {best_sharpe['roi']:.2f}%")
        
    def generate_summary_report(self):
        """Generate a comprehensive HTML summary report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Backtest Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .insight-box {{ 
            background: #e8f4f8; 
            border-left: 4px solid #007bff; 
            padding: 15px; 
            margin: 20px 0;
            border-radius: 4px;
        }}
        .metric {{ 
            display: inline-block; 
            margin: 10px 20px 10px 0; 
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .positive {{ color: #28a745; font-weight: bold; }}
        .negative {{ color: #dc3545; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Advanced Backtest Analytics Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Executive Summary</h2>
        <div class="insight-box">
            <h3>Key Insights</h3>
        """
        
        # Add key insights based on analysis
        if 'score_differential' in self.analytics_results:
            score_df = self.analytics_results['score_differential']
            best_diff = score_df['total_pnl'].idxmax()
            worst_diff = score_df['total_pnl'].idxmin()
            
            html += f"""
            <p><strong>Score Differential:</strong> 
            Best performance when score difference is {best_diff} 
            (P&L: £{score_df.loc[best_diff, 'total_pnl']:.2f}). 
            Avoid betting when difference is {worst_diff}.</p>
            """
        
        if 'odds_ranges' in self.analytics_results:
            odds_df = self.analytics_results['odds_ranges']
            best_odds = odds_df['roi'].idxmax()
            
            html += f"""
            <p><strong>Optimal Odds Range:</strong> 
            {best_odds} with ROI of {odds_df.loc[best_odds, 'roi']:.2f}%</p>
            """
        
        if 'optimization' in self.analytics_results:
            opt_df = self.analytics_results['optimization']
            best = opt_df.loc[opt_df['roi'].idxmax()]
            
            html += f"""
            <p><strong>Optimal Parameters:</strong> 
            Confidence {best['confidence']:.0f}%, 
            Kelly {best['kelly']:.2f}, 
            ROI {best['roi']:.2f}%</p>
            """
        
        html += """
        </div>
        
        <h2>Detailed Analysis</h2>
        
        <h3>1. Score Differential Analysis</h3>
        <img src="score_differential_analysis.png" alt="Score Differential Analysis">
        
        <h3>2. Odds Range Performance</h3>
        <img src="odds_range_analysis.png" alt="Odds Range Analysis">
        
        <h3>3. Model Calibration</h3>
        <img src="calibration_analysis.png" alt="Calibration Analysis">
        
        <h3>4. Parameter Optimization</h3>
        <img src="grid_search_optimization.png" alt="Grid Search Results">
        
    </div>
</body>
</html>
        """
        
        with open(ANALYTICS_DIR / 'analytics_report.html', 'w') as f:
            f.write(html)
        
        print(f"\n✅ Full analytics report saved to: {ANALYTICS_DIR / 'analytics_report.html'}")


def main():
    parser = argparse.ArgumentParser(description='Advanced Backtest Analytics')
    parser.add_argument('--confidence-range', nargs=3, type=float, 
                       default=[60, 85, 5],
                       help='Confidence range: start end step (default: 60 85 5)')
    parser.add_argument('--kelly-values', nargs='+', type=float,
                       default=[0, 0.125, 0.25, 0.5],
                       help='Kelly fractions to test (0 = flat betting)')
    parser.add_argument('--bankroll', type=float, default=1000.0,
                       help='Initial bankroll for simulations')
    parser.add_argument('--max-bet-values', nargs='+', type=float,
                       default=[0.05],
                       help='Maximum bet fractions to test')
    
    args = parser.parse_args()
    
    # Generate confidence thresholds
    conf_start, conf_end, conf_step = args.confidence_range
    confidence_thresholds = list(np.arange(conf_start, conf_end + conf_step, conf_step))
    
    print("=== Advanced Backtest Analytics ===")
    print(f"Confidence thresholds: {confidence_thresholds}")
    print(f"Kelly fractions: {args.kelly_values}")
    print(f"Max bet fractions: {args.max_bet_values}")
    print(f"Initial bankroll: £{args.bankroll}")
    
    # Run analysis
    analyzer = AdvancedBacktestAnalytics()
    
    # Run comprehensive backtests
    analyzer.run_comprehensive_backtest(
        confidence_thresholds,
        args.kelly_values,
        args.bankroll,
        args.max_bet_values
    )
    
    # Perform all analyses
    analyzer.analyze_score_differential()
    analyzer.analyze_odds_ranges()
    analyzer.analyze_calibration()
    analyzer.grid_search_optimization()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n✨ Analysis complete! Check the analytics_results folder for all outputs.")


if __name__ == '__main__':
    main()