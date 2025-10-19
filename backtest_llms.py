#!/usr/bin/env python3
"""
LLM Technical Analysis Backtesting System
Benchmarks different LLMs on their ability to perform technical analysis
"""

import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class TechnicalAnalysisBacktester:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.results = []
        
        # List of free LLMs to test from OpenRouter
        self.llms_to_test = [
            # DeepSeek models (top tier reasoning)
            "deepseek/deepseek-chat-v3.1:free",
            #"deepseek/deepseek-r1:free",
            #"deepseek/deepseek-r1-0528:free",
            
            # Qwen models (strong performance)
            "qwen/qwen3-235b-a22b:free",
            #"qwen/qwen-2.5-72b-instruct:free",
            #"qwen/qwen3-coder:free",
            
            # Meta Llama models
            "meta-llama/llama-4-maverick:free",
            #"meta-llama/llama-4-scout:free",
            #"meta-llama/llama-3.3-70b-instruct:free",
            #"meta-llama/llama-3.3-8b-instruct:free",
            
            # Google models
            "google/gemini-2.0-flash-exp:free",
            #"google/gemma-3-27b-it:free",
            #"google/gemma-3-12b-it:free",
            
            # Mistral models
            #"mistralai/mistral-small-3.2-24b-instruct:free",
            "mistralai/mistral-small-3:free",
            
            # Other interesting models
            "moonshotai/kimi-k2:free",
            "microsoft/mai-ds-r1:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "tencent/hunyuan-a13b-instruct:free",
        ]
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators"""
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        return df
    
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical stock data and calculate indicators"""
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        df = self.calculate_technical_indicators(df)
        return df
    
    def create_test_case(self, df: pd.DataFrame, index: int, lookback: int = 30) -> Dict[str, Any]:
        """Create a test case with historical data and the actual outcome"""
        if index < lookback or index >= len(df):
            return None
        
        # Get lookback period data
        historical_data = df.iloc[index-lookback:index].copy()
        
        # Get the actual outcome (next day's movement)
        current_price = df.iloc[index]['Close']
        next_price = df.iloc[index + 1]['Close'] if index + 1 < len(df) else None
        
        if next_price is None:
            return None
        
        price_change_pct = ((next_price - current_price) / current_price) * 100
        
        # Determine actual direction
        if price_change_pct > 0.5:
            actual_direction = "UP"
        elif price_change_pct < -0.5:
            actual_direction = "DOWN"
        else:
            actual_direction = "NEUTRAL"
        
        return {
            "historical_data": historical_data,
            "current_price": current_price,
            "next_price": next_price,
            "price_change_pct": price_change_pct,
            "actual_direction": actual_direction,
            "date": df.index[index].strftime('%Y-%m-%d')
        }
    
    def format_data_for_llm(self, test_case: Dict[str, Any]) -> str:
        """Format technical data into a prompt for the LLM"""
        df = test_case['historical_data']
        
        # Get recent data (last 5 days)
        recent = df.tail(5)
        
        prompt = f"""You are a technical analyst. Analyze the following stock data and predict whether the stock will go UP, DOWN, or remain NEUTRAL for the next trading day.

Current Price: ${test_case['current_price']:.2f}

RECENT PRICE ACTION (Last 5 days):
"""
        for idx, row in recent.iterrows():
            prompt += f"\n{idx.strftime('%Y-%m-%d')}: Open=${row['Open']:.2f}, High=${row['High']:.2f}, Low=${row['Low']:.2f}, Close=${row['Close']:.2f}, Volume={int(row['Volume']):,}"
        
        # Add technical indicators
        latest = df.iloc[-1]
        prompt += f"""

TECHNICAL INDICATORS (Latest):
- SMA 20: ${latest['SMA_20']:.2f}
- SMA 50: ${latest['SMA_50']:.2f}
- RSI: {latest['RSI']:.2f}
- MACD: {latest['MACD']:.4f}
- Signal Line: {latest['Signal_Line']:.4f}
- Bollinger Bands: Upper=${latest['BB_Upper']:.2f}, Middle=${latest['BB_Middle']:.2f}, Lower=${latest['BB_Lower']:.2f}
- Volume vs 20-day avg: {(latest['Volume'] / latest['Volume_SMA'] * 100):.1f}%

ANALYSIS REQUIRED:
Based on this technical data, will the stock price go UP (>0.5%), DOWN (<-0.5%), or NEUTRAL (-0.5% to 0.5%) tomorrow?

Respond with ONLY ONE WORD: UP, DOWN, or NEUTRAL
Do not include any explanation, just the prediction.
"""
        return prompt
    
    def query_llm(self, model: str, prompt: str) -> str:
        """Query an LLM via OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Low temperature for more consistent predictions
            "max_tokens": 10,
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, data=json.dumps(data), timeout=30)
            response.raise_for_status()
            result = response.json()
            
            prediction = result['choices'][0]['message']['content'].strip().upper()
            
            # Extract just the prediction word
            if 'UP' in prediction:
                return 'UP'
            elif 'DOWN' in prediction:
                return 'DOWN'
            elif 'NEUTRAL' in prediction:
                return 'NEUTRAL'
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            print(f"Error querying {model}: {e}")
            return 'ERROR'
    
    def run_backtest(self, tickers: List[str], num_test_cases: int = 20):
        """Run the backtest across multiple stocks and LLMs"""
        print(f"\n{'='*60}")
        print(f"Starting LLM Technical Analysis Backtest")
        print(f"{'='*60}\n")
        
        # Fetch data for all tickers
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        all_test_cases = []
        
        for ticker in tickers:
            try:
                df = self.fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                # Create test cases at regular intervals
                step = max(1, (len(df) - 60) // num_test_cases)
                for i in range(60, len(df) - 1, step):
                    test_case = self.create_test_case(df, i)
                    if test_case:
                        test_case['ticker'] = ticker
                        all_test_cases.append(test_case)
                        if len([tc for tc in all_test_cases if tc['ticker'] == ticker]) >= num_test_cases:
                            break
                            
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        print(f"\nGenerated {len(all_test_cases)} test cases across {len(tickers)} stocks\n")
        
        # Test each LLM
        llm_results = {}
        
        for llm in self.llms_to_test:
            print(f"\n{'='*60}")
            print(f"Testing: {llm}")
            print(f"{'='*60}\n")
            
            correct = 0
            total = 0
            predictions = []
            
            for i, test_case in enumerate(all_test_cases):
                prompt = self.format_data_for_llm(test_case)
                prediction = self.query_llm(llm, prompt)
                
                actual = test_case['actual_direction']
                is_correct = prediction == actual
                
                if prediction not in ['ERROR', 'UNKNOWN']:
                    total += 1
                    if is_correct:
                        correct += 1
                
                result = {
                    'test_case_id': i,
                    'ticker': test_case['ticker'],
                    'date': test_case['date'],
                    'current_price': test_case['current_price'],
                    'next_price': test_case['next_price'],
                    'price_change_pct': test_case['price_change_pct'],
                    'predicted': prediction,
                    'actual': actual,
                    'correct': is_correct
                }
                
                predictions.append(result)
                
                if (i + 1) % 5 == 0 or i == 0:
                    accuracy = (correct / total * 100) if total > 0 else 0
                    print(f"Progress: {i+1}/{len(all_test_cases)} | Current Accuracy: {accuracy:.1f}% | Correct: {correct}/{total}")
            
            accuracy = (correct / total * 100) if total > 0 else 0
            
            llm_results[llm] = {
                'model': llm,
                'total_predictions': total,
                'correct_predictions': correct,
                'accuracy': accuracy,
                'predictions': predictions
            }
            
            print(f"\n{llm} Results:")
            print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        return llm_results
    
    def save_results(self, results: Dict[str, Any], output_file: str = "backtest_results.json"):
        """Save results to a JSON file"""
        # Create rankings
        rankings = []
        for llm, data in results.items():
            rankings.append({
                'rank': 0,
                'model': llm,
                'accuracy': data['accuracy'],
                'correct': data['correct_predictions'],
                'total': data['total_predictions']
            })
        
        # Sort by accuracy
        rankings.sort(key=lambda x: x['accuracy'], reverse=True)
        for i, item in enumerate(rankings):
            item['rank'] = i + 1
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'rankings': rankings,
            'detailed_results': results
        }
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        output_path = f"results/{output_file}"
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"FINAL RANKINGS")
        print(f"{'='*60}\n")
        
        for item in rankings:
            print(f"{item['rank']}. {item['model']}")
            print(f"   Accuracy: {item['accuracy']:.2f}% ({item['correct']}/{item['total']})")
            print()
        
        print(f"Results saved to: {output_path}\n")
        
        return output_path


def main():
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenRouter API key.")
        print("Example: OPENROUTER_API_KEY=your_key_here")
        return
    
    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # Stocks to test
    NUM_TEST_CASES_PER_STOCK = 10  # Number of predictions per stock
    
    # Initialize backtester
    backtester = TechnicalAnalysisBacktester(api_key)
    
    # Run backtest
    results = backtester.run_backtest(TICKERS, NUM_TEST_CASES_PER_STOCK)
    
    # Save results
    backtester.save_results(results)


if __name__ == "__main__":
    main()

