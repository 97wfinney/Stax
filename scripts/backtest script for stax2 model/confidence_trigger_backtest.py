#!/usr/bin/env python3
"""
Confidence Trigger Backtester - Tests betting strategy that places a single bet
per match once confidence threshold is met. Supports flat and Kelly betting.

This script loads pre-trained Stax models and runs backtests on all matches
in the historical data folder, betting once per match when confidence exceeds
the threshold.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import warnings
import argparse
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import random
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STAX_MODEL_DIR = MODELS_DIR / "stax_kfold"
REPORT_FILE = Path(__file__).resolve().parent / "confidence_trigger_backtest_report.html"

# --- Betting Parameters ---
DEFAULT_FLAT_STAKE = 10.0
TIME_INTERVALS = list(range(10, 91, 5))  # Check every 5 minutes from 10 to 90

# Reuse the StaxModelLoader from acca.py
from acca import StaxModelLoader


class ConfidenceTriggerBacktester:
    """Backtester that places a single bet per match when confidence threshold is met."""
    
    def __init__(self, stax_model: StaxModelLoader, confidence_threshold: float, 
                 kelly_fraction: Optional[float] = None, initial_bankroll: float = 100.0,
                 max_bet_fraction: float = 0.05):
        self.stax_model = stax_model
        self.confidence_threshold = confidence_threshold
        self.kelly_fraction = kelly_fraction
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_bet_fraction = max_bet_fraction
        self.results = []
        self.random_results = []
        self.bankroll_history = [initial_bankroll]  # Track bankroll over time
        
    def calculate_kelly_stake(self, confidence: float, odds: float) -> float:
        """Calculate Kelly stake size as fraction of current bankroll."""
        # Kelly formula: f = (p*b - q) / b
        # where p = probability of winning, b = net odds (odds - 1), q = 1 - p
        p = confidence / 100.0  # Convert percentage to probability
        q = 1 - p
        b = odds - 1
        
        # Avoid division by zero and negative bets
        if b <= 0:
            return 0
        
        kelly_full = (p * b - q) / b
        
        # Apply fractional Kelly
        kelly_adjusted = kelly_full * self.kelly_fraction
        
        # Apply maximum bet constraint
        max_bet = self.current_bankroll * self.max_bet_fraction
        
        # Calculate actual bet amount
        if kelly_adjusted <= 0:
            return 0
        
        bet_fraction = min(kelly_adjusted, self.max_bet_fraction)
        bet_amount = self.current_bankroll * bet_fraction
        
        # Ensure we don't bet more than our bankroll
        bet_amount = min(bet_amount, self.current_bankroll)
        
        # Minimum bet constraint (0.1% of initial bankroll)
        min_bet = self.initial_bankroll * 0.001
        if bet_amount < min_bet:
            return 0
            
        return bet_amount
        
    def process_match(self, match_file: Path) -> Dict:
        """Process a single match file and return betting result."""
        with open(match_file, 'r') as f:
            match_data = json.load(f)
        
        match_name = match_file.stem.split('__')[0].replace('_', ' ').title()
        
        # Try to place bet at each time interval
        for time_min in TIME_INTERVALS:
            prediction_info = self.stax_model.get_prediction_for_timestep(match_data, time_min)
            
            if prediction_info is None:
                continue
                
            confidence = prediction_info['confidence'] * 100
            
            # Check if confidence threshold is met
            if confidence >= self.confidence_threshold:
                # Calculate stake
                if self.kelly_fraction is not None:
                    stake = self.calculate_kelly_stake(confidence, prediction_info['odds'])
                    if stake == 0:  # Kelly says don't bet
                        continue
                else:
                    stake = DEFAULT_FLAT_STAKE
                
                # Place bet
                is_correct = prediction_info['prediction_index'] == prediction_info['actual_outcome']
                pnl = (stake * prediction_info['odds'] - stake) if is_correct else -stake
                
                # Update bankroll if using Kelly
                if self.kelly_fraction is not None:
                    self.current_bankroll += pnl
                    self.bankroll_history.append(self.current_bankroll)
                
                result = {
                    'match': match_name,
                    'file': match_file.name,
                    'bet_time': time_min,
                    'prediction': prediction_info['prediction_text'],
                    'prediction_index': prediction_info['prediction_index'],
                    'confidence': confidence,
                    'odds': prediction_info['odds'],
                    'actual_outcome': prediction_info['actual_outcome'],
                    'is_correct': is_correct,
                    'stake': stake,
                    'pnl': pnl,
                    'bankroll_after': self.current_bankroll if self.kelly_fraction else None,
                    'league': 'EPL' if 'epl' in match_file.name else 'EFL',
                    'kelly_percentage': (stake / self.current_bankroll * 100) if self.kelly_fraction else None
                }
                
                return result
        
        # No bet placed - confidence never reached threshold or Kelly said no bet
        return {
            'match': match_name,
            'file': match_file.name,
            'bet_time': None,
            'prediction': 'No Bet',
            'confidence': 0,
            'odds': 0,
            'is_correct': None,
            'stake': 0,
            'pnl': 0,
            'bankroll_after': self.current_bankroll if self.kelly_fraction else None,
            'league': 'EPL' if 'epl' in match_file.name else 'EFL'
        }
    
    def generate_random_baseline(self, match_files: List[Path]):
        """Generate random betting baseline for comparison."""
        print("Generating random baseline...")
        
        # Reset bankroll for random simulation
        random_bankroll = self.initial_bankroll
        
        for match_file in match_files:
            with open(match_file, 'r') as f:
                match_data = json.load(f)
            
            match_name = match_file.stem.split('__')[0].replace('_', ' ').title()
            
            # Get final outcome
            df, outcome = self.stax_model.process_match_for_xgb(match_data)
            if df is None or outcome == -1:
                continue
            
            # Random prediction (0=Home, 1=Away, 2=Draw)
            random_prediction = random.randint(0, 2)
            
            # Get odds at a random time (use 45 minutes as midpoint)
            prediction_info = self.stax_model.get_prediction_for_timestep(match_data, 45)
            if prediction_info is None:
                continue
            
            # Get odds for random prediction
            odds_array = df.iloc[45 * 60 // 40][['avg_home_odds', 'avg_away_odds', 'avg_draw_odds']].values
            random_odds = odds_array[random_prediction]
            
            # Random confidence between 33% and 100%
            random_confidence = random.uniform(33, 100)
            
            # Calculate stake (use same method as main strategy)
            if self.kelly_fraction is not None:
                # For random, use uniform confidence of 40% for Kelly calculation
                stake = self.calculate_kelly_stake(40, random_odds)
                if stake == 0:
                    stake = self.initial_bankroll * 0.01  # Force small bet for comparison
            else:
                stake = DEFAULT_FLAT_STAKE
            
            is_correct = random_prediction == outcome
            pnl = (stake * random_odds - stake) if is_correct else -stake
            
            if self.kelly_fraction is not None:
                random_bankroll += pnl
            
            self.random_results.append({
                'match': match_name,
                'prediction_index': random_prediction,
                'odds': random_odds,
                'is_correct': is_correct,
                'stake': stake,
                'pnl': pnl,
                'bankroll_after': random_bankroll
            })
    
    def run_backtest(self, backtest_dir: Path):
        """Run the confidence trigger backtest on all matches."""
        # Get all JSON files
        match_files = list(backtest_dir.glob('*.json'))
        print(f"Found {len(match_files)} matches to test")
        print(f"Confidence threshold: {self.confidence_threshold}%")
        if self.kelly_fraction:
            print(f"Kelly fraction: {self.kelly_fraction}")
            print(f"Initial bankroll: £{self.initial_bankroll}")
            print(f"Max bet fraction: {self.max_bet_fraction * 100}%")
        else:
            print(f"Flat stake: £{DEFAULT_FLAT_STAKE}")
        print("-" * 50)
        
        # Process each match
        for i, match_file in enumerate(match_files):
            if (i + 1) % 20 == 0:
                print(f"Processing match {i + 1}/{len(match_files)}...")
                if self.kelly_fraction:
                    print(f"  Current bankroll: £{self.current_bankroll:.2f}")
            
            result = self.process_match(match_file)
            self.results.append(result)
        
        # Generate random baseline
        self.generate_random_baseline(match_files)
        
        print("\nBacktest complete!")
        if self.kelly_fraction:
            print(f"Final bankroll: £{self.current_bankroll:.2f}")
            print(f"Bankroll growth: {(self.current_bankroll / self.initial_bankroll - 1) * 100:.2f}%")
        
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics from results."""
        df = pd.DataFrame(self.results)
        df_bets = df[df['stake'] > 0]  # Only matches where bets were placed
        
        if len(df_bets) == 0:
            return {
                'total_matches': len(df),
                'matches_bet': 0,
                'total_stake': 0,
                'total_pnl': 0,
                'roi': 0,
                'win_rate': 0,
                'avg_confidence': 0,
                'avg_odds': 0,
                'bet_timing': {},
                'league_breakdown': {},
                'kelly_stats': None
            }
        
        # Random baseline stats
        random_df = pd.DataFrame(self.random_results)
        random_pnl = random_df['pnl'].sum()
        random_total_stake = random_df['stake'].sum()
        random_roi = (random_pnl / random_total_stake) * 100 if random_total_stake > 0 else 0
        random_win_rate = random_df['is_correct'].mean() * 100
        
        # Kelly-specific stats
        kelly_stats = None
        if self.kelly_fraction is not None:
            # Calculate max drawdown
            peak = self.initial_bankroll
            max_drawdown = 0
            for value in self.bankroll_history:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            kelly_stats = {
                'initial_bankroll': self.initial_bankroll,
                'final_bankroll': self.current_bankroll,
                'bankroll_growth': (self.current_bankroll / self.initial_bankroll - 1) * 100,
                'max_drawdown': max_drawdown,
                'avg_bet_size': df_bets['stake'].mean(),
                'max_bet_size': df_bets['stake'].max(),
                'min_bet_size': df_bets['stake'].min(),
                'avg_kelly_percentage': df_bets['kelly_percentage'].mean() if 'kelly_percentage' in df_bets else 0,
                'random_final_bankroll': self.random_results[-1]['bankroll_after'] if self.random_results else self.initial_bankroll
            }
        
        # Bet timing distribution
        bet_timing = df_bets['bet_time'].value_counts().sort_index().to_dict()
        
        # League breakdown
        league_stats = {}
        for league in ['EPL', 'EFL']:
            league_df = df_bets[df_bets['league'] == league]
            if len(league_df) > 0:
                league_stats[league] = {
                    'matches': len(league_df),
                    'pnl': league_df['pnl'].sum(),
                    'win_rate': league_df['is_correct'].mean() * 100,
                    'roi': (league_df['pnl'].sum() / league_df['stake'].sum()) * 100
                }
        
        return {
            'total_matches': len(df),
            'matches_bet': len(df_bets),
            'matches_no_bet': len(df) - len(df_bets),
            'total_stake': df_bets['stake'].sum(),
            'total_pnl': df_bets['pnl'].sum(),
            'roi': (df_bets['pnl'].sum() / df_bets['stake'].sum()) * 100,
            'win_rate': df_bets['is_correct'].mean() * 100,
            'avg_confidence': df_bets['confidence'].mean(),
            'avg_odds': df_bets['odds'].mean(),
            'best_win': df_bets[df_bets['is_correct']]['pnl'].max() if any(df_bets['is_correct']) else 0,
            'worst_loss': df_bets[~df_bets['is_correct']]['pnl'].min() if any(~df_bets['is_correct']) else 0,
            'bet_timing': bet_timing,
            'league_breakdown': league_stats,
            'random_pnl': random_pnl,
            'random_roi': random_roi,
            'random_win_rate': random_win_rate,
            'kelly_stats': kelly_stats
        }
    
    def generate_html_report(self):
        """Generate comprehensive HTML report."""
        stats = self.generate_summary_statistics()
        
        # Create bet timing chart data
        timing_labels = list(stats['bet_timing'].keys())
        timing_values = list(stats['bet_timing'].values())
        
        # Prepare bankroll history data for Kelly betting
        bankroll_chart_html = ""
        if self.kelly_fraction is not None and stats['kelly_stats']:
            bankroll_labels = list(range(len(self.bankroll_history)))
            bankroll_values = self.bankroll_history
            random_bankroll_values = [r['bankroll_after'] for r in self.random_results]
            
            bankroll_chart_html = f"""
        <div class="section">
            <h2>Bankroll Progression (Kelly Betting)</h2>
            <div class="chart-container" style="height: 400px;">
                <canvas id="bankrollChart"></canvas>
            </div>
            <div class="kelly-stats-grid">
                <div class="kelly-stat">
                    <div class="stat-label">Initial Bankroll</div>
                    <div class="stat-value">£{stats['kelly_stats']['initial_bankroll']:.2f}</div>
                </div>
                <div class="kelly-stat">
                    <div class="stat-label">Final Bankroll</div>
                    <div class="stat-value {'positive' if stats['kelly_stats']['final_bankroll'] > stats['kelly_stats']['initial_bankroll'] else 'negative'}">
                        £{stats['kelly_stats']['final_bankroll']:.2f}
                    </div>
                </div>
                <div class="kelly-stat">
                    <div class="stat-label">Growth</div>
                    <div class="stat-value {'positive' if stats['kelly_stats']['bankroll_growth'] > 0 else 'negative'}">
                        {stats['kelly_stats']['bankroll_growth']:.2f}%
                    </div>
                </div>
                <div class="kelly-stat">
                    <div class="stat-label">Max Drawdown</div>
                    <div class="stat-value negative">-{stats['kelly_stats']['max_drawdown']:.2f}%</div>
                </div>
                <div class="kelly-stat">
                    <div class="stat-label">Avg Bet Size</div>
                    <div class="stat-value">£{stats['kelly_stats']['avg_bet_size']:.2f}</div>
                </div>
                <div class="kelly-stat">
                    <div class="stat-label">Avg Kelly %</div>
                    <div class="stat-value">{stats['kelly_stats']['avg_kelly_percentage']:.2f}%</div>
                </div>
            </div>
        </div>
            """
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Confidence Trigger Backtest Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            margin: 0;
            background-color: #f4f7f9;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a2b4d 0%, #2c4270 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        }}
        .stat-value {{
            font-size: 2.2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            margin-top: 0;
            color: #1a2b4d;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .win-row {{ background-color: #d4edda; }}
        .loss-row {{ background-color: #f8d7da; }}
        .no-bet-row {{ background-color: #f8f9fa; color: #6c757d; }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        .comparison-box {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .strategy-box {{
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .model-box {{
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
        }}
        .random-box {{
            background: linear-gradient(135deg, #6c757d 0%, #868e96 100%);
            color: white;
        }}
        .kelly-stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .kelly-stat {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Confidence Trigger Backtest Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Confidence Threshold: {self.confidence_threshold}%</p>
            <p>Betting Strategy: {'Kelly (fraction=' + str(self.kelly_fraction) + ')' if self.kelly_fraction else 'Flat stake £' + str(DEFAULT_FLAT_STAKE)}</p>
            {'<p>Initial Bankroll: £' + str(self.initial_bankroll) + '</p>' if self.kelly_fraction else ''}
        </div>
        
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-label">Total Matches</div>
                <div class="stat-value neutral">{stats['total_matches']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Matches Bet</div>
                <div class="stat-value neutral">{stats['matches_bet']}</div>
                <div style="font-size: 0.9em; color: #666;">
                    ({stats['matches_bet']/stats['total_matches']*100:.1f}% of matches)
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value {'positive' if stats['total_pnl'] > 0 else 'negative'}">
                    £{stats['total_pnl']:.2f}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">ROI</div>
                <div class="stat-value {'positive' if stats['roi'] > 0 else 'negative'}">
                    {stats['roi']:.2f}%
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value neutral">{stats['win_rate']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value neutral">{stats['avg_confidence']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Odds</div>
                <div class="stat-value neutral">{stats['avg_odds']:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Staked</div>
                <div class="stat-value neutral">£{stats['total_stake']:.2f}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Model vs Random Betting Comparison</h2>
            <div class="comparison-box">
                <div class="strategy-box model-box">
                    <h3>Stax Model</h3>
                    <div style="font-size: 2em; font-weight: bold;">£{stats['total_pnl']:.2f}</div>
                    <div>ROI: {stats['roi']:.2f}%</div>
                    <div>Win Rate: {stats['win_rate']:.1f}%</div>
                    {f"<div>Final Bankroll: £{stats['kelly_stats']['final_bankroll']:.2f}</div>" if stats['kelly_stats'] else ""}
                </div>
                <div class="strategy-box random-box">
                    <h3>Random Betting</h3>
                    <div style="font-size: 2em; font-weight: bold;">£{stats['random_pnl']:.2f}</div>
                    <div>ROI: {stats['random_roi']:.2f}%</div>
                    <div>Win Rate: {stats['random_win_rate']:.1f}%</div>
                    {f"<div>Final Bankroll: £{stats['kelly_stats']['random_final_bankroll']:.2f}</div>" if stats['kelly_stats'] else ""}
                </div>
            </div>
            <p style="text-align: center; margin-top: 20px; font-style: italic;">
                Model outperformance: £{stats['total_pnl'] - stats['random_pnl']:.2f} 
                ({stats['roi'] - stats['random_roi']:.2f}% ROI difference)
            </p>
        </div>
        
        {bankroll_chart_html}
        
        <div class="section">
            <h2>Betting Time Distribution</h2>
            <div class="chart-container">
                <canvas id="timingChart"></canvas>
            </div>
        </div>
        """
        
        # Add league breakdown if available
        if stats['league_breakdown']:
            html += """
        <div class="section">
            <h2>League Performance Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>League</th>
                        <th>Matches Bet</th>
                        <th>Total P&L</th>
                        <th>ROI</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody>
            """
            for league, league_stats in stats['league_breakdown'].items():
                html += f"""
                    <tr>
                        <td><strong>{league}</strong></td>
                        <td>{league_stats['matches']}</td>
                        <td class="{'positive' if league_stats['pnl'] > 0 else 'negative'}">
                            £{league_stats['pnl']:.2f}
                        </td>
                        <td class="{'positive' if league_stats['roi'] > 0 else 'negative'}">
                            {league_stats['roi']:.2f}%
                        </td>
                        <td>{league_stats['win_rate']:.1f}%</td>
                    </tr>
                """
            html += """
                </tbody>
            </table>
        </div>
            """
        
        # Add detailed results table
        html += """
        <div class="section">
            <h2>Detailed Match Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Match</th>
                        <th>League</th>
                        <th>Bet Time (min)</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Odds</th>
                        <th>Stake</th>
                        <th>Result</th>
                        <th>P&L</th>
        """
        if self.kelly_fraction:
            html += "<th>Bankroll After</th>"
        html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        # Sort results by P&L descending
        df_results = pd.DataFrame(self.results)
        df_results_sorted = df_results.sort_values('pnl', ascending=False)
        
        for _, row in df_results_sorted.iterrows():
            if row['stake'] > 0:  # Bet was placed
                row_class = 'win-row' if row['is_correct'] else 'loss-row'
                result_text = '✅ Win' if row['is_correct'] else '❌ Loss'
                html += f"""
                    <tr class="{row_class}">
                        <td>{row['match']}</td>
                        <td>{row['league']}</td>
                        <td>{row['bet_time']}</td>
                        <td>{row['prediction']}</td>
                        <td>{row['confidence']:.1f}%</td>
                        <td>{row['odds']:.2f}</td>
                        <td>£{row['stake']:.2f}</td>
                        <td>{result_text}</td>
                        <td class="{'positive' if row['pnl'] > 0 else 'negative'}">
                            £{row['pnl']:.2f}
                        </td>"""
                if self.kelly_fraction:
                    html += f"<td>£{row['bankroll_after']:.2f}</td>"
                html += """
                    </tr>
                """
            else:  # No bet placed
                html += f"""
                    <tr class="no-bet-row">
                        <td>{row['match']}</td>
                        <td>{row['league']}</td>
                        <td>-</td>
                        <td>No Bet</td>
                        <td>< {self.confidence_threshold}%</td>
                        <td>-</td>
                        <td>£0.00</td>
                        <td>-</td>
                        <td>£0.00</td>"""
                if self.kelly_fraction:
                    html += f"<td>£{row['bankroll_after']:.2f}</td>"
                html += """
                    </tr>
                """
        
        html += f"""
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Bet timing chart
        const ctx = document.getElementById('timingChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {timing_labels},
                datasets: [{{
                    label: 'Number of Bets',
                    data: {timing_values},
                    backgroundColor: 'rgba(26, 43, 77, 0.8)',
                    borderColor: 'rgba(26, 43, 77, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            stepSize: 1
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Minutes into Match'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Distribution of Bet Placement Times'
                    }}
                }}
            }}
        }});
        """
        
        # Add bankroll chart if using Kelly
        if self.kelly_fraction is not None:
            html += f"""
        // Bankroll progression chart
        const ctx2 = document.getElementById('bankrollChart').getContext('2d');
        new Chart(ctx2, {{
            type: 'line',
            data: {{
                labels: {bankroll_labels},
                datasets: [{{
                    label: 'Model Bankroll',
                    data: {bankroll_values},
                    borderColor: 'rgba(40, 167, 69, 1)',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                }}, {{
                    label: 'Random Bankroll',
                    data: {random_bankroll_values},
                    borderColor: 'rgba(108, 117, 125, 1)',
                    backgroundColor: 'rgba(108, 117, 125, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: false,
                        title: {{
                            display: true,
                            text: 'Bankroll (£)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Bet Number'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Bankroll Progression Over Time'
                    }}
                }}
            }}
        }});
            """
        
        html += """
    </script>
</body>
</html>
        """
        
        with open(REPORT_FILE, 'w') as f:
            f.write(html)
        
        print(f"\n✅ HTML report saved to: {REPORT_FILE}")


def main():
    """Main function to run the confidence trigger backtest."""
    parser = argparse.ArgumentParser(description='Confidence Trigger Backtester')
    parser.add_argument('--confidence', type=float, default=70.0,
                        help='Minimum confidence level in percent (e.g., 70.0)')
    parser.add_argument('--kelly', type=float, default=None,
                        help='Kelly fraction (e.g., 0.25 for quarter Kelly). If not set, uses flat stakes.')
    parser.add_argument('--bank', type=float, default=100.0,
                        help='Initial bankroll (default: 100)')
    parser.add_argument('--max-bet', type=float, default=0.05,
                        help='Maximum bet as fraction of bankroll (default: 0.05 = 5%)')
    args = parser.parse_args()
    
    print("=== Confidence Trigger Backtester ===")
    print(f"Testing strategy: Bet once per match when confidence ≥ {args.confidence}%")
    if args.kelly:
        print(f"Using Kelly criterion with fraction: {args.kelly}")
        print(f"Initial bankroll: £{args.bank}")
        print(f"Maximum bet size: {args.max_bet * 100}% of bankroll\n")
    else:
        print(f"Using flat stake betting: £{DEFAULT_FLAT_STAKE}\n")
    
    # Initialize and load models
    stax_model = StaxModelLoader()
    stax_model.load_saved_models()
    
    # Initialize backtester
    backtester = ConfidenceTriggerBacktester(
        stax_model, 
        args.confidence,
        kelly_fraction=args.kelly,
        initial_bankroll=args.bank,
        max_bet_fraction=args.max_bet
    )
    
    # Run backtest
    backtest_dir = DATA_DIR / "Backtest"
    backtester.run_backtest(backtest_dir)
    
    # Generate report
    backtester.generate_html_report()
    
    # Print summary to console
    stats = backtester.generate_summary_statistics()
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total Matches: {stats['total_matches']}")
    print(f"Matches Bet: {stats['matches_bet']} ({stats['matches_bet']/stats['total_matches']*100:.1f}%)")
    print(f"Total P&L: £{stats['total_pnl']:.2f}")
    print(f"ROI: {stats['roi']:.2f}%")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    
    if args.kelly and stats['kelly_stats']:
        print(f"\n=== KELLY BETTING STATS ===")
        print(f"Initial Bankroll: £{stats['kelly_stats']['initial_bankroll']:.2f}")
        print(f"Final Bankroll: £{stats['kelly_stats']['final_bankroll']:.2f}")
        print(f"Bankroll Growth: {stats['kelly_stats']['bankroll_growth']:.2f}%")
        print(f"Max Drawdown: {stats['kelly_stats']['max_drawdown']:.2f}%")
        print(f"Average Bet Size: £{stats['kelly_stats']['avg_bet_size']:.2f}")
        print(f"Average Kelly %: {stats['kelly_stats']['avg_kelly_percentage']:.2f}%")
    
    print(f"\n=== MODEL vs RANDOM ===")
    print(f"Model P&L: £{stats['total_pnl']:.2f} (ROI: {stats['roi']:.2f}%)")
    print(f"Random P&L: £{stats['random_pnl']:.2f} (ROI: {stats['random_roi']:.2f}%)")
    print(f"Outperformance: £{stats['total_pnl'] - stats['random_pnl']:.2f}")
    
    print("\n✨ Backtest complete! Check the HTML report for detailed results.")


if __name__ == '__main__':
    main()