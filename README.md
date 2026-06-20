# Stock Market Forecasting Using Recurrent Neural Networks

A comprehensive implementation of RNN-based models for predicting technology stock prices using historical market data.

## Project Description

This project explores the application of recurrent neural networks for time series forecasting in financial markets. We analyze 12 years of historical stock data from four major technology companies and build predictive models using Simple RNN, LSTM, and GRU architectures.

## Dataset Information

**Companies Analyzed:**
- Amazon (AMZN)
- Google (GOOGL)  
- IBM
- Microsoft (MSFT)

**Data Characteristics:**
- Time Period: January 2006 - December 2017
- Trading Days: 3,019 per stock
- Features: Open, High, Low, Close, Volume
- Combined Features: 20 (5 per stock)

## Project Structure

```
stock-prediction-rnn-project/
│
├── stock_market_forecasting.ipynb    # Main analysis notebook
├── stock_data/                        # Historical price data
│   ├── AMZN_stocks_data.csv
│   ├── GOOGL_stocks_data.csv
│   ├── IBM_stocks_data.csv
│   └── MSFT_stocks_data.csv
└── README.md                          # Project documentation
```

## Methodology

### Data Preprocessing
1. **Data Integration**: Combined individual stock datasets into unified dataframe
2. **Sequence Generation**: Created sliding window sequences of 60 trading days
3. **Train-Test Split**: Chronological 80/20 split to preserve temporal order
4. **Normalization**: Applied MinMaxScaler independently per sequence to prevent data leakage

### Model Architectures

**Simple RNN**
- Vanilla recurrent architecture with tanh activation
- Tested configurations from single layer to stacked architectures
- Baseline model for comparison

**LSTM (Long Short-Term Memory)**
- Advanced architecture with gating mechanisms
- Better at capturing long-term dependencies
- Addresses vanishing gradient problem

**GRU (Gated Recurrent Unit)**
- Simplified gating compared to LSTM
- Fewer parameters, faster training
- Competitive performance with LSTM

### Hyperparameter Tuning

Systematic grid search across:
- Layer configurations: [40], [80], [120], [80, 40], [120, 60]
- Dropout rates: 0.0, 0.2
- Learning rates: 0.001, 0.0005
- Batch size: 64
- Early stopping: patience of 6-10 epochs

## Results

### Single-Stock Prediction (AMZN)

| Model | Configuration | Test MAE | Test RMSE | MAPE |
|-------|--------------|----------|-----------|------|
| Simple RNN | [40], dropout=0.0 | $XXX.XX | $XXX.XX | XX.X% |
| LSTM | [80], dropout=0.2 | $XXX.XX | $XXX.XX | XX.X% |
| GRU | [80, 40], dropout=0.0 | $XXX.XX | $XXX.XX | XX.X% |

### Multi-Stock Prediction

The multi-target models simultaneously predict closing prices for all four stocks, leveraging cross-stock correlations.

**Key Observations:**
- LSTM and GRU outperform Simple RNN on volatile stocks
- IBM shows lowest prediction error due to price stability
- High-growth stocks (AMZN, GOOGL) exhibit larger absolute errors but reasonable relative errors
- Multi-target training provides implicit regularization

## Key Insights

1. **Temporal Patterns**: 60-day sequences effectively capture quarterly business cycles
2. **Correlation Benefits**: High inter-stock correlation (0.95+ for price, 0.40+ for returns) justifies multi-stock modeling
3. **Architecture Choice**: Gating mechanisms in LSTM/GRU critical for long-sequence learning
4. **Volatility Challenge**: Prediction difficulty increases with stock volatility
5. **Market Dynamics**: Models struggle with unprecedented trends (exponential growth phases)

## Technical Requirements

```
Python 3.8+
tensorflow >= 2.21
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.4
seaborn >= 0.11
scikit-learn >= 1.0
```

## Installation and Usage

```bash
# Clone or download the project
cd stock-prediction-rnn-project

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch notebook
jupyter notebook stock_market_forecasting.ipynb
```

## Reproducibility

All models use fixed random seeds (seed=123) for reproducibility. Running the notebook will regenerate all results and visualizations.

## Limitations

- Historical data only, no incorporation of news or sentiment
- Single-step ahead prediction (t+1 forecasting)
- Sensitive to sudden market disruptions
- No transaction costs or trading constraints considered
- Past performance does not guarantee future results

## Future Enhancements

- Integration of sentiment analysis from financial news
- Attention mechanisms for interpretable predictions
- Multi-horizon forecasting (1-day, 1-week, 1-month ahead)
- Ensemble methods combining multiple architectures
- Risk quantification with prediction intervals
- Real-time prediction pipeline

## Author Notes

This project is developed for educational purposes to demonstrate RNN applications in financial time series analysis. It is not intended as investment advice.

## License

This project is available for academic and educational use.
