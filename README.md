# TimeSeriesForecast


## Primo Esperimento (S&P500)
Il primo esperimento si compone di diversi file dove sono implementate in python i vari algoritimi testati: 
1. ARIMA/ARIMAX
2. Logistic/Linear Regression
3. Hidden Markov Model con le sue differenti iterazioni

Il dataset si trova in SP500Data.csv e deriva da una semplice richiesta attraverso le api yfinance `data = yf.download(ticker, start=start_date)`

## Secondo Esperimetno (Voli Europei 2016+)
In questo secondo esperimento invece è stata realizzato il medesimo confront con metodologie differenti su un altro dataset tra:
1. ARIMA
2. SARIMA
3. ETS

Qui il dataset invece è stato scaricato in formato csv da kaggle a causa del peso non uploadabile qui su github (https://www.kaggle.com/datasets/umerhaddii/european-flights-dataset?resource=download)
