import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime


class ModelBacktester:
    """
    Comprehensive backtesting and evaluation for LSTM match prediction models.
    """
    
    def __init__(self, model_path: str, sequence_length: int, backtest_dir: str, 
                 prediction_minutes: int = 45):
        """
        Initialize backtester with trained model.
        
        Args:
            model_path: Path to saved model directory
            sequence_length: Sequence length used in training
            backtest_dir: Path to backtest data directory
            prediction_minutes: When to make prediction (default: 45 mins = halftime)
        """
        self.sequence_length = sequence_length
        self.backtest_dir = Path(backtest_dir)
        self.features = 4
        self.prediction_minutes = prediction_minutes
        self.prediction_interval = int(prediction_minutes * 60 / 40)  # Convert to intervals
        
        # Load model and scaler
        self.model = load_model(f"{model_path}/lstm_model.h5")
        self.scaler = joblib.load(f"{model_path}/scaler.pkl")
        
        # Results storage
        self.predictions = []
        self.actuals = []
        self.match_results = []
        
    def extract_features_from_match(self, match_data: List[Dict]) -> Tuple[np.ndarray, int, Dict]:
        """
        Extract features and outcome from a match.
        
        Returns:
            features: Array of shape (n_intervals, 4)
            outcome: 0=home win, 1=away win, 2=draw
            match_info: Dictionary with match details
        """
        # Limit to 180 intervals
        match_data = match_data[:180]
        
        features = []
        for entry in match_data:
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
                # Get all keys from odds dictionary
                odds_keys = list(odds.keys())
                
                # Filter out 'Draw' to find team names
                team_keys = [k for k in odds_keys if k != 'Draw']
                
                if len(team_keys) >= 2:
                    # Assume first team is home, second is away
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
        
        # Get final outcome
        final_score = match_data[-1]['score'].split(' - ')
        final_home = int(final_score[0])
        final_away = int(final_score[1])
        
        if final_home > final_away:
            outcome = 0  # Home win
        elif final_away > final_home:
            outcome = 1  # Away win
        else:
            outcome = 2  # Draw
        
        match_info = {
            'match_name': match_data[0]['match'],
            'final_score': match_data[-1]['score'],
            'total_intervals': len(match_data),
            'initial_odds': {
                'home': features[0][0],
                'away': features[0][1],
                'draw': features[0][2]
            }
        }
        
        return np.array(features), outcome, match_info
    
    def predict_match_progressive(self, features: np.ndarray) -> List[np.ndarray]:
        """
        Make predictions at different points throughout the match.
        
        Returns:
            List of probability arrays at different time points
        """
        predictions_over_time = []
        
        # Make predictions every 10 intervals (about every 6-7 minutes)
        for i in range(self.sequence_length, len(features), 10):
            sequence = features[i-self.sequence_length:i]
            
            # Normalize
            sequence_normalized = self.scaler.transform(sequence)
            sequence_normalized = sequence_normalized.reshape(1, self.sequence_length, self.features)
            
            # Predict
            pred = self.model.predict(sequence_normalized, verbose=0)
            predictions_over_time.append({
                'interval': i,
                'minute': i * 40 / 60,  # Approximate minute
                'probabilities': pred[0]
            })
        
        return predictions_over_time
    
    def evaluate_single_match(self, json_path: str) -> Dict:
        """
        Evaluate model performance on a single match.
        """
        with open(json_path, 'r') as f:
            match_data = json.load(f)
        
        # Extract features and outcome
        features, actual_outcome, match_info = self.extract_features_from_match(match_data)
        
        if len(features) < self.sequence_length:
            return None
        
        # Get predictions over time
        predictions_over_time = self.predict_match_progressive(features)
        
        # Make prediction at specified time point
        prediction_point = min(self.prediction_interval, len(features) - self.sequence_length)
        
        # For realistic prediction, use data only up to prediction point
        if prediction_point < self.sequence_length:
            return None
            
        final_sequence = features[prediction_point-self.sequence_length:prediction_point]
        final_normalized = self.scaler.transform(final_sequence)
        final_normalized = final_normalized.reshape(1, self.sequence_length, self.features)
        final_pred = self.model.predict(final_normalized, verbose=0)[0]
        
        predicted_outcome = np.argmax(final_pred)
        
        # Store for aggregate metrics
        self.predictions.append(predicted_outcome)
        self.actuals.append(actual_outcome)
        
        # Get score at prediction time
        prediction_entry = match_data[min(prediction_point, len(match_data)-1)]
        score_at_prediction = prediction_entry['score']
        
        result = {
            'match_info': match_info,
            'actual_outcome': actual_outcome,
            'predicted_outcome': predicted_outcome,
            'final_probabilities': final_pred,
            'correct': predicted_outcome == actual_outcome,
            'confidence': float(np.max(final_pred)),
            'predictions_over_time': predictions_over_time,
            'score_at_prediction': score_at_prediction,
            'prediction_minute': self.prediction_minutes,
            'probability_evolution': {
                'home': [p['probabilities'][0] for p in predictions_over_time],
                'away': [p['probabilities'][1] for p in predictions_over_time],
                'draw': [p['probabilities'][2] for p in predictions_over_time],
                'minutes': [p['minute'] for p in predictions_over_time]
            }
        }
        
        return result
    
    def run_backtest(self, max_matches: int = None) -> Dict:
        """
        Run backtest on all matches in the backtest directory.
        """
        json_files = list(self.backtest_dir.glob('*.json'))
        if max_matches:
            json_files = json_files[:max_matches]
        
        print(f"Running backtest on {len(json_files)} matches...")
        
        for i, json_file in enumerate(json_files):
            try:
                result = self.evaluate_single_match(str(json_file))
                if result:
                    self.match_results.append(result)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(json_files)} matches")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # Calculate aggregate metrics
        aggregate_results = self.calculate_aggregate_metrics()
        
        return aggregate_results
    
    def calculate_aggregate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        """
        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)
        
        # Basic metrics
        accuracy = accuracy_score(actuals, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            actuals, predictions, average=None, labels=[0, 1, 2]
        )
        
        # Confusion matrix
        cm = confusion_matrix(actuals, predictions)
        
        # Calculate profit/loss if betting on predictions
        profit_loss = self.calculate_betting_performance()
        
        # Confidence analysis
        confidence_stats = self.analyze_confidence()
        
        results = {
            'total_matches': len(self.match_results),
            'overall_accuracy': accuracy,
            'class_metrics': {
                'home_win': {
                    'precision': precision[0],
                    'recall': recall[0],
                    'f1_score': f1[0],
                    'support': int(support[0])
                },
                'away_win': {
                    'precision': precision[1],
                    'recall': recall[1],
                    'f1_score': f1[1],
                    'support': int(support[1])
                },
                'draw': {
                    'precision': precision[2],
                    'recall': recall[2],
                    'f1_score': f1[2],
                    'support': int(support[2])
                }
            },
            'confusion_matrix': cm.tolist(),
            'profit_loss': profit_loss,
            'confidence_analysis': confidence_stats,
            'prediction_distribution': {
                'home_predicted': int(np.sum(predictions == 0)),
                'away_predicted': int(np.sum(predictions == 1)),
                'draw_predicted': int(np.sum(predictions == 2))
            }
        }
        
        return results
    
    def calculate_betting_performance(self) -> Dict:
        """
        Calculate hypothetical betting performance.
        """
        total_bet = 0
        total_return = 0
        wins = 0
        
        for result in self.match_results:
            if result['correct']:
                wins += 1
                # Estimate return based on typical odds
                # This is simplified - you'd use actual odds in production
                if result['predicted_outcome'] == 0:  # Home win
                    total_return += 2.5  # Typical home odds
                elif result['predicted_outcome'] == 1:  # Away win
                    total_return += 3.5  # Typical away odds
                else:  # Draw
                    total_return += 3.0  # Typical draw odds
            
            total_bet += 1
        
        return {
            'total_bets': total_bet,
            'winning_bets': wins,
            'win_rate': wins / total_bet if total_bet > 0 else 0,
            'roi': (total_return - total_bet) / total_bet if total_bet > 0 else 0,
            'profit_loss': total_return - total_bet
        }
    
    def analyze_confidence(self) -> Dict:
        """
        Analyze model confidence vs accuracy.
        """
        confidences = [r['confidence'] for r in self.match_results]
        correct_high_conf = sum(1 for r in self.match_results 
                               if r['correct'] and r['confidence'] > 0.6)
        total_high_conf = sum(1 for r in self.match_results 
                             if r['confidence'] > 0.6)
        
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'high_confidence_accuracy': correct_high_conf / total_high_conf if total_high_conf > 0 else 0,
            'high_confidence_count': total_high_conf
        }
    
    def generate_report(self, results: Dict, output_path: str = None):
        """
        Generate a comprehensive performance report.
        """
        report = []
        report.append("="*60)
        report.append(f"LSTM Model Backtest Report - Sequence Length: {self.sequence_length}")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Matches Evaluated: {results['total_matches']}")
        report.append("")
        
        # Overall Performance
        report.append("OVERALL PERFORMANCE")
        report.append("-"*30)
        report.append(f"Accuracy: {results['overall_accuracy']:.3f}")
        report.append("")
        
        # Class-wise Performance
        report.append("CLASS-WISE METRICS")
        report.append("-"*30)
        for outcome, metrics in results['class_metrics'].items():
            report.append(f"\n{outcome.upper()}:")
            report.append(f"  Precision: {metrics['precision']:.3f}")
            report.append(f"  Recall: {metrics['recall']:.3f}")
            report.append(f"  F1-Score: {metrics['f1_score']:.3f}")
            report.append(f"  Support: {metrics['support']}")
        
        # Confusion Matrix
        report.append("\nCONFUSION MATRIX")
        report.append("-"*30)
        report.append("         Predicted")
        report.append("         Home  Away  Draw")
        labels = ['Home', 'Away', 'Draw']
        for i, row in enumerate(results['confusion_matrix']):
            report.append(f"Actual {labels[i]:4} {row[0]:4} {row[1]:4} {row[2]:4}")
        
        # Betting Performance
        report.append("\nBETTING PERFORMANCE (Hypothetical)")
        report.append("-"*30)
        pl = results['profit_loss']
        report.append(f"Win Rate: {pl['win_rate']:.3f}")
        report.append(f"ROI: {pl['roi']:.3f}")
        report.append(f"Profit/Loss: {pl['profit_loss']:.2f} units")
        
        # Confidence Analysis
        report.append("\nCONFIDENCE ANALYSIS")
        report.append("-"*30)
        ca = results['confidence_analysis']
        report.append(f"Mean Confidence: {ca['mean_confidence']:.3f}")
        report.append(f"High Confidence (>0.6) Accuracy: {ca['high_confidence_accuracy']:.3f}")
        report.append(f"High Confidence Predictions: {ca['high_confidence_count']}")
        
        # Save report
        report_text = "\n".join(report)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text
    
    def plot_performance(self, results: Dict, save_path: str = None):
        """
        Create visualization plots for model performance.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Confusion Matrix Heatmap
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Home', 'Away', 'Draw'],
                   yticklabels=['Home', 'Away', 'Draw'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Class-wise Metrics
        metrics_df = pd.DataFrame(results['class_metrics']).T
        metrics_df[['precision', 'recall', 'f1_score']].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Class-wise Performance Metrics')
        axes[0, 1].set_xlabel('Outcome')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend(['Precision', 'Recall', 'F1-Score'])
        
        # 3. Prediction Distribution
        pred_dist = results['prediction_distribution']
        axes[1, 0].pie([pred_dist['home_predicted'], 
                       pred_dist['away_predicted'], 
                       pred_dist['draw_predicted']],
                      labels=['Home', 'Away', 'Draw'],
                      autopct='%1.1f%%')
        axes[1, 0].set_title('Prediction Distribution')
        
        # 4. Example probability evolution
        if self.match_results:
            # Plot first match's probability evolution
            match = self.match_results[0]
            prob_evo = match['probability_evolution']
            axes[1, 1].plot(prob_evo['minutes'], prob_evo['home'], label='Home', linewidth=2)
            axes[1, 1].plot(prob_evo['minutes'], prob_evo['away'], label='Away', linewidth=2)
            axes[1, 1].plot(prob_evo['minutes'], prob_evo['draw'], label='Draw', linewidth=2)
            axes[1, 1].set_xlabel('Match Time (minutes)')
            axes[1, 1].set_ylabel('Probability')
            axes[1, 1].set_title('Example: Probability Evolution During Match')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Test each sequence length model at different prediction times
    sequence_lengths = [5, 10, 20]
    prediction_times = [30, 45, 60]  # Predict at 30 mins, halftime, and 60 mins
    
    for seq_len in sequence_lengths:
        for pred_time in prediction_times:
            print(f"\n{'='*60}")
            print(f"Backtesting Model - Seq Length: {seq_len}, Prediction at: {pred_time} mins")
            print(f"{'='*60}\n")
            
            # Initialize backtester
            backtester = ModelBacktester(
                model_path=f"./lstm_seq{seq_len}",
                sequence_length=seq_len,
                backtest_dir="../data/Backtest",
                prediction_minutes=pred_time
            )
            
            # Run backtest
            results = backtester.run_backtest(max_matches=None)
            
            # Generate report
            backtester.generate_report(
                results, 
                output_path=f"./backtest_report_seq{seq_len}_pred{pred_time}.txt"
            )
            
            # Only generate plots for halftime predictions
            if pred_time == 45:
                backtester.plot_performance(
                    results,
                    save_path=f"./backtest_plots_seq{seq_len}_pred{pred_time}.png"
                )
            
            print(f"\nCompleted backtest for seq_len={seq_len}, pred_time={pred_time}")
            print(f"Accuracy: {results['overall_accuracy']:.3f}")