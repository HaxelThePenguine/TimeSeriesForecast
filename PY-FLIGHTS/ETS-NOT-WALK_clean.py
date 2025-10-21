import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing # ETS
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="talk")


'''
Anche qui si potevano condensare tutti le funzioni e i modelli in un solo file
probabilmente andava anche meglio
pero per chiarezza diciamo va bene così

'''


## Analisi dei vari dati (cerca pattern, stagionalità, trend, ecc.)

def detect_seasonality_pattern(series):
    """
    Rileva pattern stagionali
    """
    print("\nRILEVAMENTO PATTERN STAGIONALI")
    print("="*50)

    if len(series) >= 14:
        try:
            decomposition_weekly = seasonal_decompose(series, model='additive', period=7, extrapolate_trend='freq')
            weekly_seasonal_var = np.var(decomposition_weekly.seasonal)
            print(f"Varianza stagionale settimanale: {weekly_seasonal_var:.0f}")
        except:
            weekly_seasonal_var = 0
    else:
        weekly_seasonal_var = 0

    if len(series) >= 60:
        try:
            decomposition_monthly = seasonal_decompose(series, model='additive', period=30, extrapolate_trend='freq')
            monthly_seasonal_var = np.var(decomposition_monthly.seasonal)
            print(f"Varianza stagionale mensile: {monthly_seasonal_var:.0f}")
        except:
            monthly_seasonal_var = 0
    else:
        monthly_seasonal_var = 0

    if len(series) >= 730:
        try:
            decomposition_yearly = seasonal_decompose(series, model='additive', period=365, extrapolate_trend='freq')
            yearly_seasonal_var = np.var(decomposition_yearly.seasonal)
            print(f"Varianza stagionale annuale: {yearly_seasonal_var:.0f}")
        except:
            yearly_seasonal_var = 0
    else:
        yearly_seasonal_var = 0

    seasonal_vars = {
        7: weekly_seasonal_var,
        30: monthly_seasonal_var,
        365: yearly_seasonal_var
    }

    best_period = max(seasonal_vars, key=seasonal_vars.get)
    if seasonal_vars[best_period] > series.var() * 0.01:
        print(f"Pattern stagionale rilevato: periodo {best_period}")
        return best_period
    else:
        print("Nessun pattern stagionale significativo rilevato")
        return None


# Selezione automatica modello ETS

def auto_ets_model_selection(series, seasonal_period=None):
    """
    Selezione automatica modello ETS ottimale
    """
    print("\nSELEZIONE AUTOMATICA MODELLO ETS")
    print("="*50)

    models_to_test = []

    models_to_test.extend([
        {'trend': None, 'seasonal': None, 'name': 'Simple Exponential Smoothing'},
        {'trend': 'add', 'seasonal': None, 'name': 'Holt Linear Trend'},
        {'trend': 'mul', 'seasonal': None, 'name': 'Holt Exponential Trend'}
    ])
    seasonal_period =7
    if seasonal_period and seasonal_period <= len(series) // 2:
        models_to_test.extend([
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': seasonal_period, 'name': 'Holt-Winters Additive'},
            {'trend': 'add', 'seasonal': 'mul', 'seasonal_periods': seasonal_period, 'name': 'Holt-Winters Multiplicative'},
            {'trend': None, 'seasonal': 'add', 'seasonal_periods': seasonal_period, 'name': 'Seasonal Additive'},
            {'trend': None, 'seasonal': 'mul', 'seasonal_periods': seasonal_period, 'name': 'Seasonal Multiplicative'}
        ])

    best_aic = np.inf
    best_model = None
    best_fitted = None
    aic_results = []

    print(f"Testing {len(models_to_test)} modelli ETS...")

    # test vari modelli ETS
    for model_config in models_to_test:
        try:
            trend = model_config.get('trend')
            seasonal = model_config.get('seasonal')
            seasonal_periods = model_config.get('seasonal_periods')
            name = model_config['name']

            if seasonal and seasonal_periods:
                model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal,
                                           seasonal_periods=seasonal_periods)
            else:
                model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal)

            fitted_model = model.fit(optimized=True, remove_bias=True)

            aic = fitted_model.aic
            aic_results.append({
                'name': name,
                'trend': trend,
                'seasonal': seasonal,
                'seasonal_periods': seasonal_periods,
                'aic': aic
            })

            # confronta AIC per selezionare il migliore
            if aic < best_aic:
                best_aic = aic
                best_model = model_config
                best_fitted = fitted_model

            print(f"   {name}: AIC = {aic:.2f}")

        except Exception as e:
            print(f"   {model_config['name']}: Failed ({str(e)[:50]})")
            continue

    if best_fitted is None:
        print("FALLBACK: Uso Simple Exponential Smoothing")
        model = ExponentialSmoothing(series, trend=None, seasonal=None)
        best_fitted = model.fit()
        best_model = {'name': 'Simple Exponential Smoothing (Fallback)'}
        best_aic = best_fitted.aic

    print(f"\nMODELLO OTTIMALE: {best_model['name']} con AIC = {best_aic:.2f}")

    return best_fitted, best_model, best_aic, aic_results

def fit_ets_and_forecast(series, ets_model, forecast_periods=18):
    """
    Fit ETS e generazione previsioni
    """
    print(f"\nFIT MODELLO ETS")
    print("="*50)

    try:
        print(f"Modello fittato con successo")
        print(f"AIC: {ets_model.aic:.2f}")

        if hasattr(ets_model, 'params'):
            if hasattr(ets_model.params, 'smoothing_level'):
                print(f"Alpha (livello): {ets_model.params.smoothing_level:.4f}")
            if hasattr(ets_model.params, 'smoothing_trend') and ets_model.params.smoothing_trend:
                print(f"Beta (trend): {ets_model.params.smoothing_trend:.4f}")
            if hasattr(ets_model.params, 'smoothing_seasonal') and ets_model.params.smoothing_seasonal:
                print(f"Gamma (stagionale): {ets_model.params.smoothing_seasonal:.4f}")

        forecast_values = ets_model.forecast(steps=forecast_periods)

        residuals = series - ets_model.fittedvalues
        residual_std = np.std(residuals)

        conf_lower = forecast_values - 1.96 * residual_std
        conf_upper = forecast_values + 1.96 * residual_std

        conf_int = pd.DataFrame({
            'lower': conf_lower,
            'upper': conf_upper
        }, index=forecast_values.index)

        print(f"Previsioni generate per {forecast_periods} periodi futuri")
        print(f"Deviazione standard residui: {residual_std:.0f} voli")

        return {
            'fitted_model': ets_model,
            'forecast_values': forecast_values,
            'confidence_intervals': conf_int,
            'residuals': residuals,
            'aic': ets_model.aic,
            'residual_std': residual_std
        }

    except Exception as e:
        print(f"Errore nel fit del modello: {str(e)}")
        return None

## solite metriche e visulizzazioni  
def calculate_forecast_metrics(actual, predicted):
    """
    Calcola metriche di previsione
    """
    print("\nCALCOLO METRICHE DI PREVISIONE")
    print("="*50)

    mse_insample = mean_squared_error(actual, predicted[:len(actual)])
    mae_insample = mean_absolute_error(actual, predicted[:len(actual)])
    rmse_insample = np.sqrt(mse_insample)
    mape_insample = np.mean(np.abs((actual - predicted[:len(actual)]) / actual)) * 100

    r2_insample = r2_score(actual, predicted[:len(actual)])

    print(f"METRICHE IN-SAMPLE:")
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
    }

## no visualizzazioni

def main():
    print("ETS SEMPLICE - VOLI EUROPEI 2016+")
    print("="*60)

    file_path = 'DATA/flights.csv'

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
    data = data.rename(columns={value_column: 'Close'})

    data['Date'] = data[date_column].dt.date
    aggregated_data = data.groupby('Date')['Close'].sum().reset_index()
    aggregated_data['Date'] = pd.to_datetime(aggregated_data['Date'])
    aggregated_data.set_index('Date', inplace=True)

    data = aggregated_data.sort_index()
    data = data.dropna()

    print(f"Dataset caricato: {len(data)} osservazioni")
    print(f"Periodo: {data.index[0].strftime('%Y-%m-%d')} → {data.index[-1].strftime('%Y-%m-%d')}")


    seasonal_period = detect_seasonality_pattern(data['Close'])



    train_size = len(data) - 12
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"\nDIVISIONE DATI :")

    print(f"Training: {len(train_data)} obs ({train_data.index[0].strftime('%Y-%m-%d')} → {train_data.index[-1].strftime('%Y-%m-%d')})")
    if len(test_data) > 0:
        print(f"Test: {len(test_data)} obs ({test_data.index[0].strftime('%Y-%m-%d')} → {test_data.index[-1].strftime('%Y-%m-%d')})")

    optimal_model, model_info, best_aic, aic_results = auto_ets_model_selection(train_data['Close'], seasonal_period)

    forecast_periods = len(test_data) 
    forecast_result = fit_ets_and_forecast(train_data['Close'], optimal_model, forecast_periods)

    if forecast_result is None:
        print("Impossibile fittare il modello ETS")
        return

    fitted_values = forecast_result['fitted_model'].fittedvalues
    metrics = calculate_forecast_metrics(train_data['Close'], fitted_values)

    # confrontoo con test set 
    if len(test_data) > 0:
        forecast_values = forecast_result['forecast_values']
        n_test = min(len(test_data), len(forecast_values))

        print(f"\nCONFRONTO PREVISIONI ETS vs TEST SET REALE:")
        print("="*50)

        test_mse = mean_squared_error(test_data['Close'].iloc[:n_test], forecast_values.iloc[:n_test])
        test_mae = mean_absolute_error(test_data['Close'].iloc[:n_test], forecast_values.iloc[:n_test])
        test_mape = np.mean(np.abs((test_data['Close'].iloc[:n_test] - forecast_values.iloc[:n_test]) / test_data['Close'].iloc[:n_test])) * 100
        test_r2 = r2_score(test_data['Close'].iloc[:n_test], forecast_values.iloc[:n_test])

        print(f"METRICHE OUT-OF-SAMPLE (Test Set):")
        print(f"   R² Score: {test_r2:.4f}")
        print(f"   MSE: {test_mse:.0f} voli²")
        print(f"   RMSE: {np.sqrt(test_mse):.0f} voli")
        print(f"   MAE: {test_mae:.0f} voli")
        print(f"   MAPE: {test_mape:.2f}%")

        print(f"\nCONFRONTO DETTAGLIATO (primi 5 giorni):")
        for i in range(min(5, n_test)):
            real_val = test_data['Close'].iloc[i]
            pred_val = forecast_values.iloc[i]
            error = abs(real_val - pred_val)
            error_pct = (error / real_val) * 100
            date_str = test_data.index[i].strftime('%Y-%m-%d')
            print(f"   {date_str}: Reale={real_val:.0f}, Previsto={pred_val:.0f}, Errore={error:.0f} ({error_pct:.1f}%)")

        metrics['test_r2'] = test_r2
        metrics['test_mse'] = test_mse
        metrics['test_rmse'] = np.sqrt(test_mse)
        metrics['test_mae'] = test_mae
        metrics['test_mape'] = test_mape
        metrics['test_size'] = n_test
        metrics['test_data'] = test_data['Close'].iloc[:n_test]
        metrics['test_forecasts'] = forecast_values.iloc[:n_test]
        metrics['test_dates'] = test_data.index[:n_test]

    #visualize_ets_results(data, forecast_result, model_info, metrics)

    #save_results_to_csv(data, forecast_result)

    print(f"\n{'='*60}")
    print(f"ETS {model_info['name']} VOLI EUROPEI COMPLETATO")
    print(f"AIC del modello: {forecast_result['aic']:.2f}")
    print(f"R² in-sample: {metrics['r2_insample']:.4f}")
    print(f"RMSE in-sample: {metrics['rmse_insample']:.0f} voli")
    if 'test_r2' in metrics:
        print(f"R² out-of-sample: {metrics['test_r2']:.4f}")
        print(f"RMSE out-of-sample: {metrics['test_rmse']:.0f} voli")

if __name__ == "__main__":
    main()