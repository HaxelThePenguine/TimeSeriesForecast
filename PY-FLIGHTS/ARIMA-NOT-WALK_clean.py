import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA # importazione ARIMA libreria top
from statsmodels.tsa.stattools import adfuller 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="talk")

'''
Anche qui si potevano condensare tutti le funzioni e i modelli in un solo file
probabilmente andava anche meglio
pero per chiarezza diciamo va bene così

'''



#parte di logica ARIMA
def test_stationarity_detailed(series, max_d=2):
    """
    Test di stazionarietà 
    """
    print("\nTEST DI STAZIONARIETÀ")


    result_original = adfuller(series.dropna())
    print(f"Serie originale:")
    print(f"   ADF Statistic: {result_original[0]:.4f}")
    print(f"   p-value: {result_original[1]:.6f}")
    print(f"   Stazionaria: {'SI' if result_original[1] < 0.05 else 'NO'}")

    results = {'original': {'adf_stat': result_original[0], 'p_value': result_original[1],
                           'is_stationary': result_original[1] < 0.05}}

    current_series = series.copy()
    for d in range(1, max_d + 1):
        current_series = current_series.diff().dropna()
        if len(current_series) > 0:
            result_diff = adfuller(current_series)
            results[f'diff_{d}'] = {'adf_stat': result_diff[0], 'p_value': result_diff[1],
                                   'is_stationary': result_diff[1] < 0.05}
            print(f"Differenziazione {d}:")
            print(f"   ADF Statistic: {result_diff[0]:.4f}")
            print(f"   p-value: {result_diff[1]:.6f}")
            print(f"   Stazionaria: {'SI' if result_diff[1] < 0.05 else 'NO'}")

            if result_diff[1] < 0.05:
                return d

    print("Serie non stazionaria anche con 2 differenziazioni")
    return 2

def auto_arima_order_selection(series, max_p=3, max_q=3, d=None):
    """
    Selezione automatica ordini ARIMA
    """
    print("\nSELEZIONE AUTOMATICA ORDINI ARIMA")


    if d is None:
        d = test_stationarity_detailed(series)

    best_aic = np.inf
    best_order = (1, d, 1)
    aic_results = []

    print(f"Testing ordini ARIMA con d = {d}...")

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                aic_results.append({'order': (p, d, q), 'aic': aic})

                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)

                print(f"   ARIMA{p,d,q}: AIC = {aic:.2f}")

            except Exception as e:
                print(f"   ARIMA{p,d,q}: Failed")
                continue

    print(f"\nMODELLO OTTIMALE: ARIMA{best_order} con AIC = {best_aic:.2f}")

    return best_order, best_aic, aic_results

def fit_arima_and_forecast(series, order, forecast_periods=18):
    """
    Fit ARIMA e generazione previsioni
    """
    print(f"\nFIT MODELLO ARIMA{order}")
    print("-"*50)

    try:
        model = ARIMA(series, order=order)
        fitted_model = model.fit()

        print(f"Modello fittato con successo")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")

        forecast_result = fitted_model.get_forecast(steps=forecast_periods)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()

        residuals = fitted_model.resid

        print(f"Previsioni generate per {forecast_periods} periodi futuri")

        return {
            'fitted_model': fitted_model,
            'forecast_values': forecast_values,
            'confidence_intervals': conf_int,
            'residuals': residuals,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }

    except Exception as e:
        print(f"Errore nel fit del modello: {str(e)}")
        return None


## metriche e visulizzazioni un pò lunghino (molto migliorabile) ma va bene così 
def calculate_forecast_metrics(actual, predicted, predicted_t, test_data=None):
    """
    Calcola metriche di previsione
    """
    print("\nCALCOLO METRICHE DI PREVISIONE")
    print("-"*50)
    n = min(len(actual), len(predicted))
    mse_insample = mean_squared_error(actual, predicted[:len(actual)])
    mae_insample = mean_absolute_error(actual, predicted[:len(actual)])
    rmse_insample = np.sqrt(mse_insample)
    mape_insample = np.mean(np.abs((actual - predicted[:len(actual)]) / actual)) * 100

    r2_insample = r2_score(actual, predicted[:len(actual)])
    residuals = actual - predicted[:len(actual)]
    ljung_val = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].values[0]

    if test_data is not None:
        
        test_size = min(len(test_data), len(predicted_t))
        test_forecasts = predicted_t.iloc[:test_size] if hasattr(predicted_t, 'iloc') else predicted_t[:test_size]
        test_actual = test_data.iloc[:test_size] if hasattr(test_data, 'iloc') else test_data[:test_size]
        
        
        if hasattr(test_actual, 'values'):
            test_actual = test_actual.values.flatten()
        if hasattr(test_forecasts, 'values'):
            test_forecasts = test_forecasts.values.flatten()

        mse_outsample = mean_squared_error(test_actual, test_forecasts)
        mae_outsample = mean_absolute_error(test_actual, test_forecasts)
        rmse_outsample = np.sqrt(mse_outsample)
        mape_outsample = np.mean(np.abs((test_actual - test_forecasts) / test_actual)) * 100
        r2_outsample = r2_score(test_actual, test_forecasts)
        print(f"\nMETRICHE IN-SAMPLE:")
        print(f"   R² Score: {r2_insample:.4f}")
        print(f"   MSE: {mse_insample:.0f} voli²")
        print(f"   RMSE: {rmse_insample:.0f} voli")
        print(f"   MAE: {mae_insample:.0f} voli")
        print(f"   MAPE: {mape_insample:.2f}%")
        print(f"\nMETRICHE OUT-OF-SAMPLE:")
        print(f"   R² Score: {r2_outsample:.4f}")
        print(f"   MSE: {mse_outsample:.0f} voli²")
        print(f"   RMSE: {rmse_outsample:.0f} voli")
        print(f"   MAE: {mae_outsample:.0f} voli")
        print(f"   MAPE: {mape_outsample:.2f}%")

        return {
            'r2_insample': r2_insample,
            'mse_insample': mse_insample,
            'rmse_insample': rmse_insample,
            'mae_insample': mae_insample,
            'mape_insample': mape_insample,
            'ljung_val': ljung_val,
            'test_r2': r2_outsample,
            'test_mse': mse_outsample,
            'test_rmse': rmse_outsample,
            'test_mae': mae_outsample,
            'test_mape': mape_outsample,
            'test_size': test_size,
            'test_data': test_actual,
            'test_forecasts': test_forecasts,
            'test_dates': test_data.index[:test_size]
        }
    else:
        print(f"\nMETRICHE IN-SAMPLE:")
        print(f"   R² Score: {r2_insample:.4f}")
        print(f"   MSE: {mse_insample:.0f} voli²")
        print(f"   RMSE: {rmse_insample:.0f} voli")
        print(f"   MAE: {mae_insample:.0f} voli")
        print(f"   MAPE: {mape_insample:.2f}%")
        return {
            'r2_insample': r2_insample,
            'mse_insample': mse_insample,
            'rmse_insample': rmse_insample,
            'mae_insample': mae_insample,
            'mape_insample': mape_insample,
            'ljung_val': ljung_val
        }
## no visualizzazioni

def main():
    print("ARIMA SEMPLICE - VOLI EUROPEI 2016+")
    print("-"*20)

    file_path = 'c:/Users/ThePenguine/Desktop/Python test/DATA/flights.csv'

    data = pd.read_csv(file_path)

    print(f"Colonne disponibili: {list(data.columns)}")
    print(f"Prime 3 righe:")
    print(data.head(3))

    # casino per il parsing delle colonne valori e date
    if 'FLT_DATE' in data.columns and 'FLT_TOT_1' in data.columns:
        print("Rilevato file voli europei standard")
        date_column = 'FLT_DATE'
        value_column = 'FLT_TOT_1'
    else:
        date_column = None
        for col in data.columns:
            if col.lower() in ['date', 'data', 'datetime', 'timestamp', 'time', 'flt_date']:
                date_column = col
                break

        if date_column is None:
            date_column = data.columns[0]
            print(f"Usando prima colonna come data: {date_column}")

        value_column = None
        for col in data.columns:
            if col.lower() in ['close', 'voli', 'flights', 'value', 'volume', 'count', 'flt_tot_1']:
                value_column = col
                break

        if value_column is None:
            value_column = data.columns[1] if len(data.columns) > 1 else data.columns[0]
            print(f"Usando colonna come valore: {value_column}")

    print(f"Colonna data selezionata: {date_column}")
    print(f"Colonna valore selezionata: {value_column}")

    try:
        if 'T' in str(data[date_column].iloc[0]):
            data[date_column] = pd.to_datetime(data[date_column], format='%Y-%m-%dT%H:%M:%SZ')
        else:
            data[date_column] = pd.to_datetime(data[date_column], infer_datetime_format=True)
    except:
        try:
            data[date_column] = pd.to_datetime(data[date_column], format='%Y-%m-%d')
        except:
            try:
                data[date_column] = pd.to_datetime(data[date_column], format='%d/%m/%Y')
            except:
                print("Formato data non riconosciuto")
                return

    data = data[data[value_column].notna()]
    data = data[data[value_column] >= 0]
    data = data.rename(columns={value_column: 'Close'}) # rinomina colonna valori in 'Close' per coerenza chiusura giornata con i voli tot

    data['Date'] = data[date_column].dt.date
    aggregated_data = data.groupby('Date')['Close'].sum().reset_index()
    aggregated_data['Date'] = pd.to_datetime(aggregated_data['Date'])
    aggregated_data.set_index('Date', inplace=True)

    data = aggregated_data.sort_index()
    data = data.dropna()

    print(f"Dataset caricato: {len(data)} osservazioni")
    print(f"Periodo: {data.index[0].strftime('%Y-%m-%d')} → {data.index[-1].strftime('%Y-%m-%d')}")

    # Divisione dati in train e test (ultimi 12 GIORNI per test)
    train_size = len(data) - 12
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"\nDIVISIONE DATI :")

    optimal_order, best_aic, aic_results = auto_arima_order_selection(train_data['Close'])

    forecast_periods = len(test_data)

    forecast_result = fit_arima_and_forecast(train_data['Close'], optimal_order, forecast_periods)

    ## controllo mio dati perchè mi sembravano strani ma sono effetivamente così
    #forecast_result['forecast_values'] = forecast_result['forecast_values'].round().astype(int)
    #print(forecast_result['forecast_values'], "\n")
    #print(test_data['Close'], "\n")
    #R2= r2_score(test_data['Close'], forecast_result['forecast_values'])
    #print(f"R² Score out-of-sample: {R2:.4f}")

    fitted_values = forecast_result['fitted_model'].fittedvalues
    print(fitted_values)
    metrics_out_sample = calculate_forecast_metrics(train_data['Close'], fitted_values, forecast_result['forecast_values'], test_data)

 
    #visualize_arima_results(data, forecast_result, optimal_order, metrics_out_sample)

    #save_results_to_csv(data, forecast_result)


if __name__ == "__main__":
    main()