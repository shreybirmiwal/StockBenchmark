# LLM Technical Analysis Backtesting

Benchmark different Large Language Models (LLMs) on their ability to perform technical analysis and predict stock movements using **free models from OpenRouter**.

## Overview

This project fetches historical stock data from Yahoo Finance, calculates technical indicators, and asks various LLMs to predict price movements based on technical analysis. The results are ranked by accuracy.

## Features

- **Technical Indicators**: Calculates SMA, EMA, MACD, RSI, Bollinger Bands, and volume indicators
- **20 Free LLMs**: Tests models from DeepSeek, Qwen, Meta, Google, Mistral, and more
- **Comprehensive Results**: Outputs detailed JSON with predictions, actual outcomes, and rankings
- **Configurable**: Easy to adjust stocks, number of test cases, and lookback periods
- **Cost-Free**: Uses only free tier models from OpenRouter

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get OpenRouter API Key

1. Go to [OpenRouter.ai](https://openrouter.ai/keys)
2. Create an account and generate an API key (free tier available)
3. Create a `.env` file in the project root:

```bash
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

### 3. Run the Backtest

```bash
python3 backtest_llms.py
```

### 4. Visualize Results

```bash
python3 visualize_results.py
```

This will generate comprehensive charts and graphs in the `visualizations/` folder.

## Important Notes

- **Rate Limiting**: Free tier models have rate limits. The script includes automatic retry logic with exponential backoff and delays
- **Runtime**: Testing 20 models takes 15-30 minutes depending on rate limits
- **Adjustable**: Edit `NUM_TEST_CASES_PER_STOCK` in `main()` to reduce test cases if needed
- **API Key**: Keep your `.env` file safe - it's automatically ignored by git

## Configuration

Edit the `main()` function in `backtest_llms.py` to customize:

```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # Stocks to test
NUM_TEST_CASES_PER_STOCK = 5  # Predictions per stock (reduced to avoid rate limits)
```

You can also modify the list of LLMs in the `TechnicalAnalysisBacktester.__init__()` method.

## Output

Results are saved to `results/backtest_results.json` with the following structure:

```json
{
  "timestamp": "2025-10-19T...",
  "rankings": [
    {
      "rank": 1,
      "model": "deepseek/deepseek-r1:free",
      "accuracy": 72.5,
      "correct": 29,
      "total": 40
    },
    ...
  ],
  "detailed_results": {
    "model_name": {
      "predictions": [
        {
          "ticker": "AAPL",
          "date": "2024-10-15",
          "current_price": 150.23,
          "next_price": 152.11,
          "predicted": "UP",
          "actual": "UP",
          "correct": true
        },
        ...
      ]
    }
  }
}
```

## Technical Indicators Used

- **Moving Averages**: SMA 20, SMA 50, EMA 12, EMA 26
- **MACD**: Moving Average Convergence Divergence with Signal Line
- **RSI**: Relative Strength Index (14-period)
- **Bollinger Bands**: 20-period with 2 standard deviations
- **Volume Analysis**: Comparison with 20-day average

## LLMs Tested (All Free Tier)

### DeepSeek Models (Reasoning Specialists)
1. DeepSeek Chat V3.1
2. DeepSeek R1
3. DeepSeek R1-0528

### Qwen Models (Large Scale)
4. Qwen3 235B
5. Qwen 2.5 72B Instruct
6. Qwen3 Coder

### Meta Llama Models
7. Llama 4 Maverick
8. Llama 4 Scout
9. Llama 3.3 70B Instruct
10. Llama 3.3 8B Instruct

### Google Models
11. Gemini 2.0 Flash Experimental
12. Gemma 3 27B
13. Gemma 3 12B

### Mistral Models
14. Mistral Small 3.2 24B
15. Mistral Small 3

### Other Models
16. Moonshot Kimi K2
17. Microsoft MAI DS R1
18. NVIDIA Nemotron Nano 9B V2
19. Tencent Hunyuan A13B

## How It Works

1. **Data Collection**: Fetches 1 year of historical data for selected stocks
2. **Test Case Generation**: Creates test cases with 30-day lookback periods
3. **Technical Analysis**: Calculates indicators for each time window
4. **LLM Prediction**: Presents data to each LLM for prediction (UP/DOWN/NEUTRAL)
5. **Evaluation**: Compares predictions to actual price movements
6. **Ranking**: Sorts models by accuracy

## Classification

- **UP**: Price increase > 0.5%
- **DOWN**: Price decrease < -0.5%
- **NEUTRAL**: Price change between -0.5% and 0.5%

## Notes

- Uses a 30-day lookback period for each prediction
- Low temperature (0.1) for consistent predictions
- Results directory is created automatically
- All models are free tier from OpenRouter
- API rate limits may apply depending on usage

## Example Output

```
FINAL RANKINGS
============================================================

1. deepseek/deepseek-r1:free
   Accuracy: 68.00% (34/50)

2. qwen/qwen3-235b-a22b:free
   Accuracy: 66.00% (33/50)

3. google/gemini-2.0-flash-exp:free
   Accuracy: 64.00% (32/50)
...
```

## Visualizations

The `visualize_results.py` script generates the following charts:

1. **Model Rankings** - Bar chart showing accuracy for each model
2. **Confusion Matrices** - Prediction accuracy breakdown (UP/DOWN/NEUTRAL)
3. **Per-Stock Accuracy** - How each model performs on different stocks
4. **Prediction Distribution** - What types of predictions each model makes
5. **Price Movements** - Actual price changes with prediction correctness
6. **Accuracy vs Predictions** - Scatter plot comparing models
7. **Summary Report** - Text file with detailed statistics

All visualizations are saved as high-resolution PNG files in the `visualizations/` directory.

## License

MIT
