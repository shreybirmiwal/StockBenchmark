#!/usr/bin/env python3
"""
Visualization script for LLM backtesting results
Creates comprehensive charts and graphs from the JSON output
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class BacktestVisualizer:
    def __init__(self, results_file: str, exclude_models=None):
        """Load results from JSON file"""
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        # Filter out excluded models
        if exclude_models is None:
            exclude_models = ['gpt-oss', 'oss']  # Exclude GPT-OSS by default
        
        self.rankings = [r for r in self.data['rankings'] 
                        if not any(excl.lower() in r['model'].lower() for excl in exclude_models)]
        
        self.detailed_results = {k: v for k, v in self.data['detailed_results'].items()
                                if not any(excl.lower() in k.lower() for excl in exclude_models)}
        
        self.timestamp = self.data['timestamp']
        
        # Create output directory
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loaded results from {results_file}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Number of models: {len(self.rankings)}")
        if exclude_models:
            print(f"Excluded models containing: {', '.join(exclude_models)}")
    
    def plot_model_rankings(self):
        """Bar chart of model accuracy rankings"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        models = [r['model'] for r in self.rankings]
        accuracies = [r['accuracy'] for r in self.rankings]
        
        # Color bars by performance
        colors = ['#2ecc71' if acc >= 40 else '#f39c12' if acc >= 30 else '#e74c3c' 
                  for acc in accuracies]
        
        bars = ax.barh(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add accuracy labels on bars
        for i, (bar, acc, rank_data) in enumerate(zip(bars, accuracies, self.rankings)):
            width = bar.get_width()
            label = f"{acc:.1f}% ({rank_data['correct']}/{rank_data['total']})"
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   label, ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('LLM Technical Analysis Performance Rankings', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, max(accuracies) + 15)
        
        # Add reference line at 50%
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=2, 
                   label='50% (Random)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_rankings.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'model_rankings.png'}")
        plt.close()
    
    def plot_confusion_matrices(self):
        """Create confusion matrices for each model"""
        n_models = len(self.detailed_results)
        
        # Create subplots
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, (model_name, model_data) in enumerate(self.detailed_results.items()):
            predictions = model_data['predictions']
            
            # Create confusion matrix data
            classes = ['UP', 'DOWN', 'NEUTRAL']
            matrix = np.zeros((3, 3))
            
            for pred in predictions:
                actual = pred['actual']
                predicted = pred['predicted']
                
                if actual in classes and predicted in classes:
                    actual_idx = classes.index(actual)
                    pred_idx = classes.index(predicted)
                    matrix[actual_idx][pred_idx] += 1
            
            # Plot heatmap
            ax = axes[idx]
            sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes,
                       ax=ax, cbar=True, square=True)
            
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            
            # Shorten model name for title
            short_name = model_name.split('/')[-1]
            accuracy = model_data['accuracy']
            ax.set_title(f'{short_name}\n{accuracy:.1f}% accuracy', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices - Actual vs Predicted', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'confusion_matrices.png'}")
        plt.close()
    
    def plot_per_stock_accuracy(self):
        """Accuracy breakdown by stock for each model"""
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        # Calculate per-stock accuracy for each model
        stock_accuracies = {}
        
        for model_name, model_data in self.detailed_results.items():
            stock_accuracies[model_name] = {}
            
            for stock in stocks:
                stock_preds = [p for p in model_data['predictions'] if p['ticker'] == stock]
                if stock_preds:
                    correct = sum(1 for p in stock_preds if p['correct'])
                    total = len([p for p in stock_preds if p['predicted'] not in ['ERROR', 'UNKNOWN']])
                    accuracy = (correct / total * 100) if total > 0 else 0
                    stock_accuracies[model_name][stock] = accuracy
                else:
                    stock_accuracies[model_name][stock] = 0
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(stocks))
        width = 0.15
        
        for idx, (model_name, accuracies) in enumerate(stock_accuracies.items()):
            values = [accuracies[stock] for stock in stocks]
            short_name = model_name.split('/')[-1][:20]  # Shorten name
            ax.bar(x + idx * width, values, width, label=short_name, alpha=0.8)
        
        ax.set_xlabel('Stock', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy by Stock', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(stock_accuracies) - 1) / 2)
        ax.set_xticklabels(stocks)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Random')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_stock_accuracy.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'per_stock_accuracy.png'}")
        plt.close()
    
    def plot_prediction_distribution(self):
        """Distribution of prediction types for each model"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        prediction_counts = {}
        
        for model_name, model_data in self.detailed_results.items():
            predictions = [p['predicted'] for p in model_data['predictions']]
            counter = Counter(predictions)
            short_name = model_name.split('/')[-1][:25]
            prediction_counts[short_name] = {
                'UP': counter.get('UP', 0),
                'DOWN': counter.get('DOWN', 0),
                'NEUTRAL': counter.get('NEUTRAL', 0)
            }
        
        # Create stacked bar chart
        models = list(prediction_counts.keys())
        up_counts = [prediction_counts[m]['UP'] for m in models]
        down_counts = [prediction_counts[m]['DOWN'] for m in models]
        neutral_counts = [prediction_counts[m]['NEUTRAL'] for m in models]
        
        x = np.arange(len(models))
        
        p1 = ax.barh(x, up_counts, label='UP', color='#2ecc71', alpha=0.8)
        p2 = ax.barh(x, down_counts, left=up_counts, label='DOWN', color='#e74c3c', alpha=0.8)
        p3 = ax.barh(x, neutral_counts, left=[u+d for u,d in zip(up_counts, down_counts)], 
                    label='NEUTRAL', color='#95a5a6', alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(models)
        ax.set_xlabel('Number of Predictions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Type Distribution by Model', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'prediction_distribution.png'}")
        plt.close()
    
    def plot_price_movements(self):
        """Visualize actual price movements vs predictions for best model"""
        # Get best model
        best_model = self.rankings[0]['model']
        model_data = self.detailed_results[best_model]
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 16))
        
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        for idx, stock in enumerate(stocks):
            ax = axes[idx]
            stock_preds = [p for p in model_data['predictions'] if p['ticker'] == stock]
            stock_preds = sorted(stock_preds, key=lambda x: x['date'])
            
            dates = [p['date'] for p in stock_preds]
            price_changes = [p['price_change_pct'] for p in stock_preds]
            correct = [p['correct'] for p in stock_preds]
            
            # Color by correctness
            colors = ['#2ecc71' if c else '#e74c3c' for c in correct]
            
            ax.bar(range(len(dates)), price_changes, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.axhline(y=0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=-0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_title(f'{stock} - Price Changes (Green = Correct Prediction)', 
                        fontweight='bold')
            ax.set_ylabel('Price Change (%)', fontweight='bold')
            ax.set_xlabel('Test Case', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add accuracy for this stock
            stock_accuracy = sum(correct) / len(correct) * 100 if correct else 0
            ax.text(0.02, 0.98, f'Accuracy: {stock_accuracy:.1f}%', 
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Price Movements Analysis - {best_model.split("/")[-1]}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'price_movements.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'price_movements.png'}")
        plt.close()
    
    def plot_accuracy_vs_predictions(self):
        """Scatter plot of accuracy vs number of predictions"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = [r['model'].split('/')[-1][:25] for r in self.rankings]
        accuracies = [r['accuracy'] for r in self.rankings]
        totals = [r['total'] for r in self.rankings]
        
        scatter = ax.scatter(totals, accuracies, s=200, alpha=0.6, 
                           c=accuracies, cmap='RdYlGn', edgecolors='black', linewidth=2)
        
        # Add labels
        for i, model in enumerate(models):
            ax.annotate(model, (totals[i], accuracies[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Number of Valid Predictions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Number of Predictions', fontsize=14, fontweight='bold', pad=20)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Random (50%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.colorbar(scatter, ax=ax, label='Accuracy (%)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_vs_predictions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'accuracy_vs_predictions.png'}")
        plt.close()
    
    def create_summary_report(self):
        """Generate a text summary report"""
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LLM TECHNICAL ANALYSIS BACKTEST - SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Backtest Date: {self.timestamp}\n\n")
            
            f.write("RANKINGS\n")
            f.write("-"*70 + "\n")
            for rank in self.rankings:
                f.write(f"{rank['rank']:2d}. {rank['model']:<50s} "
                       f"{rank['accuracy']:6.2f}% ({rank['correct']:2d}/{rank['total']:2d})\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("DETAILED STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            for model_name, model_data in self.detailed_results.items():
                f.write(f"\n{model_name}\n")
                f.write("-"*70 + "\n")
                f.write(f"Overall Accuracy: {model_data['accuracy']:.2f}%\n")
                f.write(f"Correct: {model_data['correct_predictions']} / {model_data['total_predictions']}\n")
                
                # Per-stock breakdown
                f.write("\nPer-Stock Performance:\n")
                stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
                for stock in stocks:
                    stock_preds = [p for p in model_data['predictions'] if p['ticker'] == stock]
                    correct = sum(1 for p in stock_preds if p['correct'])
                    total = len([p for p in stock_preds if p['predicted'] not in ['ERROR', 'UNKNOWN']])
                    acc = (correct / total * 100) if total > 0 else 0
                    f.write(f"  {stock}: {acc:6.2f}% ({correct}/{total})\n")
                
                f.write("\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("="*70 + "\n\n")
            
            best = self.rankings[0]
            worst = self.rankings[-1]
            
            f.write(f"• Best Model: {best['model']} ({best['accuracy']:.2f}%)\n")
            f.write(f"• Worst Model: {worst['model']} ({worst['accuracy']:.2f}%)\n")
            f.write(f"• Average Accuracy: {np.mean([r['accuracy'] for r in self.rankings]):.2f}%\n")
            f.write(f"• Models above 50%: {sum(1 for r in self.rankings if r['accuracy'] > 50)}\n")
            f.write(f"• Models below 50%: {sum(1 for r in self.rankings if r['accuracy'] <= 50)}\n")
        
        print(f"✓ Saved: {report_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*70)
        print("Generating Visualizations")
        print("="*70 + "\n")
        
        self.plot_model_rankings()
        self.plot_confusion_matrices()
        self.plot_per_stock_accuracy()
        self.plot_prediction_distribution()
        self.plot_price_movements()
        self.plot_accuracy_vs_predictions()
        self.create_summary_report()
        
        print("\n" + "="*70)
        print(f"All visualizations saved to: {self.output_dir}/")
        print("="*70 + "\n")


def main():
    # Find the most recent results file
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("Error: 'results' directory not found.")
        print("Please run backtest_llms.py first to generate results.")
        return
    
    results_files = list(results_dir.glob('*.json'))
    
    if not results_files:
        print("Error: No results files found in 'results' directory.")
        return
    
    # Use most recent or specified file
    results_file = sorted(results_files, key=lambda x: x.stat().st_mtime)[-1]
    
    print(f"\nUsing results file: {results_file}")
    
    # Create visualizer and generate all charts
    visualizer = BacktestVisualizer(results_file)
    visualizer.generate_all_visualizations()
    
    print("\n✨ Visualization complete!")
    print(f"📊 Check the '{visualizer.output_dir}' folder for all charts and reports.\n")


if __name__ == "__main__":
    main()

