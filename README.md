# TimeSeriesForecast

A Python-based research project for **time-series analysis, forecasting, and financial market regime modeling**, developed as the experimental component of my Bachelor's thesis in Computer Engineering at the University of Modena and Reggio Emilia.

**Bachelor's Thesis:** *Time Series Analysis and Forecasting Methods*  
**Academic Year:** 2024/2025

The project investigates how different statistical and machine-learning models behave when applied to two substantially different forecasting problems:

1. **Directional forecasting of S&P 500 returns**
2. **Forecasting European daily flight counts**

The objective is to compare different modeling assumptions, evaluate their out-of-sample behavior, and study how forecasting performance changes across datasets with very different statistical properties.

---

## Overview

Time-series forecasting problems can vary significantly depending on the structure of the underlying data.

Financial markets are noisy, non-stationary, and affected by latent regimes and external factors. Conversely, datasets such as European flight traffic exhibit much stronger seasonal and structural patterns.

For this reason, the project is divided into two experiments.

### Experiment I — S&P 500

Comparison of:

- Hidden Markov Models
- ARIMA
- ARIMAX
- Linear Regression
- Logistic Regression

with a focus on **out-of-sample directional forecasting**.

### Experiment II — European Flights

Comparison of:

- ARIMA
- SARIMA
- ETS / Exponential Smoothing

with a focus on **short-term numerical forecasting in a strongly seasonal time series**.

---

# 1. S&P 500 Directional Forecasting

The first experiment studies the predictability of S&P 500 returns.

Rather than attempting only to predict the exact future value of the index, the main objective is to determine whether a model can correctly identify the **direction of the next movement**.

The problem can therefore be expressed as:

```text
Positive return  -> UP
Negative return  -> DOWN
```

The main evaluation metric is **Directional Accuracy**.

---

## Models

The following approaches were implemented and compared.

### Hidden Markov Models

Several Hidden Markov Model configurations were tested to investigate the existence of latent market regimes.

The experiments include:

- 2-state HMM
- 3-state HMM
- alternative HMM configurations using multiple Gaussian return components

The intuition is that financial markets may evolve through different hidden states, such as:

- low-volatility regimes
- high-volatility regimes
- different return-distribution environments

These regimes cannot be observed directly but can be inferred probabilistically from the historical return series.

Weekly data are used for the HMM experiments in order to reduce short-term noise and make persistent regimes easier to identify.

---

### ARIMA

ARIMA models are used to capture autoregressive relationships and dependencies between current observations and previous forecasting errors.

The implementation includes:

- automatic stationarity analysis
- differencing when required
- restricted search over `(p, d, q)` configurations
- model selection using AIC

The differencing order is dynamically selected using the **Augmented Dickey-Fuller test**.

Because ARIMA parameter estimation must be repeated during walk-forward analysis, the parameter search is restricted in order to keep the computational cost manageable.

---

### ARIMAX

ARIMAX extends ARIMA by introducing external explanatory variables.

The experiment incorporates additional financial and macroeconomic information such as:

- VIX
- U.S. Treasury yields
- Brent crude oil
- inflation-related data

The objective is to determine whether variables external to the S&P 500 itself provide additional predictive information.

---

### Linear Regression

Linear regression is applied using variables derived from historical market behavior.

Examples include:

- historical returns
- moving averages
- rolling variance
- trend-related indicators

The predicted numerical return is converted into a directional forecast according to its sign.

```text
predicted return > 0 -> UP
predicted return < 0 -> DOWN
```

---

### Logistic Regression

Logistic regression directly models the binary directional problem.

The target variable is therefore represented as:

```text
1 -> positive return
0 -> negative return
```

This allows the model to estimate the probability of a positive or negative future market movement.

---

# Walk-Forward Analysis

A central part of the financial experiment is the use of **Walk-Forward Analysis (WFA)**.

Traditional random train/test splitting is inappropriate for financial time series because it can introduce information from the future into the training process.

Instead, observations are kept strictly chronological.

A simplified walk-forward procedure is:

```text
Historical training window
        |
        v
Train model
        |
        v
Predict next observation
        |
        v
Move the window forward
        |
        v
Retrain model
        |
        v
Predict again
```

The procedure is repeated throughout the evaluation period.

For several models, a rolling training window of approximately **two years of historical observations** is used.

This produces a sequence of predictions that are generated using only information available at the time each forecast would have been made.

---

# Financial Data

S&P 500 market data are retrieved using the `yfinance` Python package.

A basic download request follows the form:

```python
import yfinance as yf

data = yf.download(ticker, start=start_date)
```

The resulting market dataset is stored locally as:

```text
SP500Data.csv
```

Depending on the model, the experiments use either:

- logarithmic returns
- percentage returns
- daily observations
- weekly observations

Additional financial and macroeconomic variables are aligned with the market time series when required by the model.

---

# Directional Accuracy

The main metric used in the financial experiment is **Directional Accuracy**.

It measures the proportion of observations for which the predicted direction matches the actual market direction.

Conceptually:

```text
Directional Accuracy =
    Correct UP/DOWN Predictions
    ---------------------------
        Total Predictions
```

A random binary prediction has an expected accuracy of approximately:

```text
50%
```

The purpose of the experiment is therefore to investigate whether any model can consistently produce an out-of-sample directional accuracy above this baseline.

---

# S&P 500 Results

The following results were obtained in the thesis experiments.

| Model | Cumulative Directional Accuracy | 2025 Accuracy | UP Accuracy |
|---|---:|---:|---:|
| HMM — 2 states | 53.97% | 53.66% | 59.32% |
| HMM — 3 states | 52.91% | 48.78% | 56.70% |
| Advanced HMM — 2 states | **55.07%** | **58.54%** | 57.96% |
| ARIMA | 52.08% | 52.88% | 54.77% |
| ARIMAX | 52.23% | 52.86% | 54.48% |
| Logistic Regression | 52.16% | 57.59% | 55.07% |
| Linear Regression | 51.00% | 51.31% | 54.96% |

The strongest cumulative result was obtained by the **advanced two-state HMM**, with approximately:

```text
55.1% cumulative directional accuracy
```

The experiment suggests that modeling latent regimes and richer return distributions may capture information that is difficult to represent using purely linear approaches.

However, these results must be interpreted carefully.

A directional forecasting advantage does **not automatically imply a profitable trading strategy**.

The experiment does not fully model several real-world effects, including:

- transaction costs
- bid-ask spreads
- slippage
- market impact
- execution latency
- taxes and brokerage fees
- liquidity constraints
- portfolio construction
- position sizing

For this reason, the results should be interpreted as an investigation of **statistical predictability**, rather than evidence of an immediately exploitable trading strategy.

---

# 2. European Flights Forecasting

The second experiment investigates a very different forecasting problem.

Instead of predicting financial returns, the objective is to forecast the **number of daily flights in Europe**.

The dataset contains European flight traffic from 2016 onward and exhibits several characteristics that are substantially different from financial data:

- strong seasonality
- recurring temporal patterns
- lower short-term stochasticity
- long-term structural trends
- the major structural shock produced by the COVID-19 pandemic

This provides an interesting comparison with the much noisier S&P 500 experiment.

---

## Models

Three classical forecasting approaches are compared.

### ARIMA

ARIMA provides a standard autoregressive forecasting baseline.

Model parameters are selected by evaluating candidate configurations and minimizing information criteria such as AIC.

---

### SARIMA

SARIMA extends the standard ARIMA framework with explicit seasonal components.

The model introduces additional seasonal autoregressive, differencing, and moving-average parameters.

This makes it particularly appropriate for the flight dataset, where recurring seasonal patterns are clearly present.

---

### ETS

ETS — Error, Trend, Seasonal — models the series through combinations of:

- level
- trend
- seasonality
- forecast error

The implementation uses exponential smoothing techniques and evaluates alternative trend and seasonal configurations.

---

# European Flights Dataset

The dataset used in this experiment is available on Kaggle:

[European Flights Dataset](https://www.kaggle.com/datasets/umerhaddii/european-flights-dataset?resource=download)

Because of the size of the original dataset, it may need to be downloaded separately before running the corresponding experiment.

The time series is constructed using aggregated daily flight counts.

---

# Train/Test Methodology

Unlike the S&P 500 experiment, the European flights experiment uses a conventional chronological train/test split.

The final:

```text
12 days
```

are reserved as the out-of-sample test period.

The models are trained on the preceding historical observations and then used to forecast the final period.

Because the test window is intentionally short, the experiment should be interpreted primarily as a **very-short-term forecasting comparison**.

---

# Evaluation Metrics

The flight forecasting experiment is evaluated using several standard regression metrics.

### Mean Absolute Percentage Error — MAPE

Measures the average percentage deviation between predictions and actual observations.

Lower values indicate better forecasting performance.

### Root Mean Squared Error — RMSE

Measures the magnitude of prediction errors while penalizing large errors more strongly.

### Mean Absolute Error — MAE

Measures the average absolute difference between predicted and observed values.

### R-Squared — R²

Measures the fraction of observed variability explained by the forecasting model.

---

# European Flights Results

The final results were:

| Model | MAPE | R² | RMSE |
|---|---:|---:|---:|
| ETS | 2.25% | 0.56 | 1349 |
| ARIMA | 3.43% | 0.083 | 1957 |
| SARIMA | **1.98%** | **0.71** | **1081** |

SARIMA achieved the strongest overall performance.

The result is consistent with the strong seasonal structure of the dataset: explicitly modeling seasonality substantially improves forecasting quality compared with standard ARIMA.

ETS also produced relatively strong results, while standard ARIMA was less capable of reproducing the seasonal dynamics of the series.

---

# Comparison Between the Two Experiments

The two experiments highlight an important aspect of time-series analysis:

> There is no universally optimal forecasting model.

Model performance strongly depends on the statistical structure of the underlying phenomenon.

### Financial Markets

S&P 500 returns are characterized by:

- substantial noise
- weak predictable structure
- volatility clustering
- latent market regimes
- external macroeconomic influences
- non-stationarity

As a consequence, even relatively small improvements over random directional prediction can be difficult to obtain consistently.

### European Flight Traffic

Flight traffic exhibits:

- strong seasonality
- recurring patterns
- more stable temporal structure
- identifiable trends

Models explicitly designed to capture these characteristics, such as SARIMA, therefore achieve substantially stronger numerical forecasting accuracy.

---

# Technologies

## Programming

- Python

## Data Processing

- NumPy
- pandas

## Statistical Modeling

- statsmodels
- ARIMA
- ARIMAX
- SARIMA
- ETS

## Machine Learning

- scikit-learn
- hmmlearn
- Linear Regression
- Logistic Regression
- Hidden Markov Models

## Data Retrieval

- yfinance

## Visualization

- matplotlib

---

# Installation

Clone the repository:

```bash
git clone https://github.com/HaxelThePenguine/TimeSeriesForecast.git
cd TimeSeriesForecast
```

The main Python dependencies used throughout the experiments include:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels hmmlearn yfinance
```

Depending on the individual script and dataset, additional packages may be required.

---

# Running the Experiments

The repository contains separate implementations for the different forecasting approaches.

The experiments should be run independently depending on the model being studied.

For the S&P 500 experiment, ensure that the required market data are available locally or retrieve them through `yfinance`.

For the European flights experiment, download the dataset from Kaggle before running the corresponding scripts.

Because different models use different preprocessing procedures, sampling frequencies, and training windows, each implementation should be considered an independent experiment rather than part of a single production forecasting pipeline.

---

# Methodological Topics Covered

The project explores several concepts related to statistical learning and time-series forecasting:

- stationarity
- differencing
- autocorrelation
- ACF and PACF
- regression
- maximum likelihood
- information criteria
- AIC / BIC
- exponential smoothing
- autoregressive models
- volatility modeling
- latent-state models
- Hidden Markov Models
- walk-forward validation
- out-of-sample testing
- directional forecasting
- exogenous variables
- seasonal forecasting
- model comparison

---

# Limitations and Future Work

The experiments provide several directions for further development.

## Financial Forecasting

Possible extensions include:

- richer feature engineering
- additional macroeconomic variables
- alternative volatility estimators
- sentiment-based variables
- improved HMM state selection
- regime-dependent forecasting models
- hyperparameter optimization
- transaction-cost-aware backtesting
- portfolio construction
- risk-adjusted performance evaluation
- cross-asset information
- higher-frequency market data

A particularly interesting extension would be to investigate whether latent market regimes can be combined with specialized forecasting models, where a different predictive model is activated depending on the inferred market state.

---

## Computational Improvements

Several models, particularly ARIMA/ARIMAX during walk-forward analysis, require repeated parameter optimization.

Future implementations could investigate:

- parallel parameter searches
- multiprocessing
- GPU-accelerated numerical methods
- more efficient rolling-window computation
- vectorized data pipelines
- distributed experimentation

These improvements would allow larger parameter spaces and datasets to be explored.

---

## Model Extensions

Additional research directions include:

- Gaussian Mixture Models
- more advanced Hidden Markov Models
- state-space models
- Bayesian time-series models
- ensemble forecasting
- LSTM / GRU architectures
- Transformer-based forecasting
- time-series foundation models

The purpose of these extensions would not simply be to introduce more complex models, but to evaluate whether their additional complexity produces measurable improvements under rigorous out-of-sample validation.

---

# Thesis

This repository contains the experimental work developed for my Bachelor's thesis:

> **Time Series Analysis and Forecasting Methods**

Bachelor's Degree in Computer Engineering  
University of Modena and Reggio Emilia  
Academic Year 2024/2025

The thesis provides the theoretical background for the statistical and machine-learning methods implemented in this repository and discusses their assumptions, strengths, limitations, and experimental results.

---

# Disclaimer

The financial forecasting experiments contained in this repository are intended exclusively for **academic and research purposes**.

The reported forecasting results should not be interpreted as financial advice or as evidence of a profitable trading strategy.

Real trading systems are affected by many factors that are outside the scope of this project, including transaction costs, execution quality, liquidity, market impact, risk management, and changing market conditions.

---

# Author

**Rocco Martino**

Computer Engineer  
M.Sc. Cybersecurity — Politecnico di Torino
B.Sc. in Computer Engineering - UniMoRe


- Cybersecurity

GitHub: [HaxelThePenguine](https://github.com/HaxelThePenguine)
