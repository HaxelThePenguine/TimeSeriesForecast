# TimeSeriesForecast

Python research project developed as the experimental component of my Bachelor's thesis in Computer Engineering:

**Time Series Analysis and Forecasting Methods**  
University of Modena and Reggio Emilia — Academic Year 2024/2025

The project compares statistical and machine-learning approaches on two different forecasting problems:

1. **Directional forecasting of S&P 500 returns**
2. **Forecasting European daily flight counts**

The main goal was to study how different models behave under different statistical properties and evaluate them using chronological out-of-sample testing.

---

## 1. S&P 500 Directional Forecasting

The first experiment investigates whether historical market data can provide useful information about the direction of future S&P 500 returns.

### Models

- Hidden Markov Models
  - 2-state HMM
  - 3-state HMM
  - multi-Gaussian return modeling
- ARIMA
- ARIMAX
- Linear Regression
- Logistic Regression

The experiments were implemented in Python using:

- NumPy
- pandas
- statsmodels
- scikit-learn
- hmmlearn
- matplotlib
- yfinance

### Data

Market data were retrieved through `yfinance`.

```python
data = yf.download(ticker, start=start_date)
```

Depending on the model, the experiments used daily or weekly S&P 500 returns. HMMs were trained on weekly observations to reduce short-term noise and identify more persistent latent market regimes.

Some ARIMAX experiments also incorporated exogenous variables such as:

- VIX
- U.S. Treasury yields
- Brent crude oil
- inflation



### Walk-Forward Evaluation

The financial models were evaluated using **walk-forward analysis**, preserving the chronological structure of the data and preventing future observations from leaking into training.

A rolling historical window is repeatedly used to:

```text
Train model
    ↓
Predict next observation
    ↓
Move window forward
    ↓
Retrain
```

For several models, the rolling training window corresponds to approximately two years of historical observations. 

The main evaluation metric is **Directional Accuracy**.

---

## S&P 500 Results

| Model | Cumulative Accuracy | 2025 Accuracy | UP Accuracy |
|---|---:|---:|---:|
| HMM — 2 states | 53.97% | 53.66% | 59.32% |
| HMM — 3 states | 52.91% | 48.78% | 56.70% |
| Advanced HMM — 2 states | **55.07%** | **58.54%** | 57.96% |
| ARIMA | 52.08% | 52.88% | 54.77% |
| ARIMAX | 52.23% | 52.86% | 54.48% |
| Logistic Regression | 52.16% | 57.59% | 55.07% |
| Linear Regression | 51.00% | 51.31% | 54.96% |

The best result was obtained by the **advanced two-state HMM**, reaching approximately **55.1% cumulative directional accuracy**. 

These results should be interpreted as evidence of limited statistical predictability rather than as a complete trading strategy. Transaction costs, spread, slippage, market impact and execution constraints were outside the scope of the experiment. 

---

# 2. European Flights Forecasting

The second experiment applies classical forecasting methods to European daily flight counts from 2016 onward.

Unlike financial returns, this dataset exhibits:

- strong seasonality
- recurring temporal patterns
- structural trends
- the COVID-19 shock

The following models were compared:

- ARIMA
- SARIMA
- ETS / Exponential Smoothing

The final 12 days were reserved as an out-of-sample test set. 

## Results

| Model | MAPE | R² | RMSE |
|---|---:|---:|---:|
| ETS | 2.25% | 0.56 | 1349 |
| ARIMA | 3.43% | 0.083 | 1957 |
| SARIMA | **1.98%** | **0.71** | **1081** |

SARIMA achieved the best forecasting performance, benefiting from its explicit modeling of the strong seasonal structure of the series. 

---

# Main Limitation: Data

The most important limitation of the financial experiment was ultimately **data quality and granularity**.

The available Yahoo Finance data were mainly daily observations and therefore provided only a coarse representation of market dynamics.

For serious short-horizon or intraday research, the model itself is only part of the problem: the underlying information set becomes critical.

A significantly stronger continuation of this work would therefore use higher-resolution market data such as:

- **1-minute OHLCV**
- tick-level trades and quotes
- Level 2 / order-book data
- ideally **Market-by-Order (MBO)** data

MBO data would make it possible to study not only price dynamics but also individual orders, queue evolution, cancellations, executions and liquidity changes.

This would enable a transition from relatively coarse daily forecasting toward more realistic research on:

- short-horizon alpha
- order-flow dynamics
- liquidity
- volatility
- market microstructure
- latent intraday regimes

In this sense, improving the **dataset and information resolution** is likely more important than simply replacing the existing models with increasingly complex architectures.

---

# Future Work

Beyond higher-quality data, possible extensions include:

- improved HMM and regime-switching models
- richer exogenous variables
- transaction-cost-aware backtesting
- ensemble models
- neural time-series models
- intraday forecasting
- order-flow and market-microstructure signals
- parallelized model selection and backtesting

The natural next step is therefore to combine more rigorous statistical modeling with substantially richer market data.

---

# Technologies

**Python:** NumPy, pandas, scikit-learn, statsmodels, hmmlearn, matplotlib  
**Models:** HMM, ARIMA, ARIMAX, SARIMA, ETS, Linear/Logistic Regression  
**Data:** yfinance, CSV

---

# Thesis

**Time Series Analysis and Forecasting Methods**  
Bachelor's Degree in Computer Engineering  
University of Modena and Reggio Emilia  
Academic Year 2024/2025

---

# Author

**Rocco Martino**

Computer Engineer  
M.Sc. Cybersecurity — Politecnico di Torino

Interests: Quantitative Research, Machine Learning, Market Microstructure, Low-Level Systems and FPGA Design

GitHub: [HaxelThePenguine](https://github.com/HaxelThePenguine)
