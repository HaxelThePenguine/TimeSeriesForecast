import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.seasonal import seasonal_decompose 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="talk")

def analyze_time_series(data):
    """
    Analisi esplorativa della serie storica
    """
    print("ANALISI ESPLORATIVA SERIE STORICA VOLI EUROPEI")
    print("="*60)

    print(f"Periodo: {data.index[0]} → {data.index[-1]}")
    print(f"Osservazioni totali: {len(data)}")
    print(f"Valore minimo: {data['Close'].min():.0f} voli")
    print(f"Valore massimo: {data['Close'].max():.0f} voli")
    print(f"Media: {data['Close'].mean():.0f} voli")
    print(f"Deviazione standard: {data['Close'].std():.0f} voli")

    total_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100

    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2021-12-31')

    pre_covid_data = data[data.index < covid_start]
    covid_data = data[(data.index >= covid_start) & (data.index <= covid_end)]
    post_covid_data = data[data.index > covid_end]

    if len(pre_covid_data) > 0 and len(covid_data) > 0:
        covid_impact = ((covid_data['Close'].min() / pre_covid_data['Close'].mean()) - 1) * 100
        print(f"Impatto COVID-19: {covid_impact:.1f}% (calo traffico)")

    if len(post_covid_data) > 0 and len(pre_covid_data) > 0:
        recovery = ((post_covid_data['Close'].mean() / pre_covid_data['Close'].mean()) - 1) * 100
        print(f"Recovery post-COVID: {recovery:.1f}% vs pre-COVID")

    print(f"Variazione totale periodo: {total_change:.1f}%")

    if len(data) > 12:
        monthly_avg = data.groupby(data.index.month)['Close'].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        print(f"Picco stagionale: Mese {peak_month} ({monthly_avg[peak_month]:.0f} voli)")
        print(f"Minimo stagionale: Mese {low_month} ({monthly_avg[low_month]:.0f} voli)")

    return {
        'total_observations': len(data),
        'min_value': data['Close'].min(),
        'max_value': data['Close'].max(),
        'mean_value': data['Close'].mean(),
        'std_value': data['Close'].std(),
        'total_change': total_change,
        'has_covid_impact': len(covid_data) > 0
    }

def test_stationarity_detailed(series, max_d=2):
    """
    Test di stazionarietà dettagliato
    """
    print("\nTEST DI STAZIONARIETÀ")
    print("="*50)

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
    print("="*50)

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
    print("="*50)

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


def calculate_forecast_metrics(actual, predicted, forecast_periods_list=[3, 6, 12]):
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
    residuals = actual - predicted[:len(actual)]
    ljung_val = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].values[0]

    print(f"METRICHE IN-SAMPLE:")
    print(f"   R² Score: {r2_insample:.4f}")
    print(f"   MSE: {mse_insample:.0f} voli²")
    print(f"   RMSE: {rmse_insample:.0f} voli")
    print(f"   MAE: {mae_insample:.0f} voli")
    print(f"   MAPE: {mape_insample:.2f}%")
    print(f"   Ljung-Box p-value: {ljung_val:.4f}")

    return {
        'r2_insample': r2_insample,
        'mse_insample': mse_insample,
        'rmse_insample': rmse_insample,
        'mae_insample': mae_insample,
        'mape_insample': mape_insample,
        'ljung_val': ljung_val
    }

def visualize_arima_results(data, forecast_result, order, metrics):
    """
    Visualizzazione risultati ARIMA
    """
    print("\nGENERAZIONE VISUALIZZAZIONI")
    print("="*50)

    fitted_model = forecast_result['fitted_model']
    forecast_values = forecast_result['forecast_values']
    conf_int = forecast_result['confidence_intervals']
    residuals = forecast_result['residuals']

    forecast_dates = forecast_values.index

    fig, axes = plt.subplots(3, 3, figsize=(24, 16))
    fig.suptitle(f'ARIMA{order} Analysis - Voli Europei 2016+ (con Close-up Previsioni)',
                 fontsize=16, fontweight='bold')

    axes[0,0].plot(data.index, data['Close'], 'b-', linewidth=2, label='Dati Storici')

    train_size = len(fitted_model.fittedvalues)
    train_dates = data.index[:train_size]
    axes[0,0].plot(train_dates, fitted_model.fittedvalues, 'r--', linewidth=1,
                   alpha=0.8, label='Fitted Values (Training)')

    if 'test_data' in metrics and 'test_forecasts' in metrics:
        test_dates = metrics['test_dates']
        axes[0,0].plot(test_dates, metrics['test_data'], 'g-', linewidth=3,
                       label=f'Test Set Reale ({len(test_dates)} mesi)', marker='o', markersize=4)

        axes[0,0].plot(test_dates, metrics['test_forecasts'], 'orange', linewidth=2,
                       linestyle='--', label='Previsioni ARIMA', marker='s', markersize=3)

        axes[0,0].fill_between(test_dates, metrics['test_data'], metrics['test_forecasts'],
                              color='red', alpha=0.3, label='Errore Previsione')
    else:
        axes[0,0].plot(forecast_dates, forecast_values, 'g-', linewidth=2, label='Previsioni ARIMA')
        axes[0,0].fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                              color='green', alpha=0.2, label='95% CI')

    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2021-12-31')
    if data.index[0] <= covid_start <= data.index[-1]:
        axes[0,0].axvspan(covid_start, covid_end, alpha=0.15, color='red', label='Periodo COVID')

    axes[0,0].set_title(f'ARIMA{order} - Previsioni vs Test Set', fontweight='bold')
    axes[0,0].set_ylabel('Numero Voli')
    axes[0,0].legend(fontsize=9)
    axes[0,0].grid(True, alpha=0.3)

    train_dates = data.index[:len(residuals)]
    axes[0,1].plot(train_dates, residuals, 'purple', alpha=0.7, linewidth=1)
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0,1].set_title('Residui del Modello', fontweight='bold')
    axes[0,1].set_ylabel('Residui')
    axes[0,1].grid(True, alpha=0.3)

    from scipy import stats
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[0,2])
    axes[0,2].set_title('Q-Q Plot Residui', fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)

    if 'test_data' in metrics and 'test_forecasts' in metrics:
        test_dates = metrics['test_dates']
        test_actual = metrics['test_data']
        test_pred = metrics['test_forecasts']

        context_months = 6
        train_end_idx = len(data) - len(test_actual)
        context_start = max(0, train_end_idx - context_months)

        context_dates = data.index[context_start:train_end_idx]
        context_values = data['Close'].iloc[context_start:train_end_idx]

        axes[1,0].plot(context_dates, context_values, 'b-', linewidth=2,
                       label='Training (contesto)', alpha=0.7)
        axes[1,0].plot(test_dates, test_actual, 'g-', linewidth=3,
                       label='Test Set Reale', marker='o', markersize=6)
        axes[1,0].plot(test_dates, test_pred, 'orange', linewidth=2,
                       linestyle='--', label='Previsioni ARIMA', marker='s', markersize=5)

        axes[1,0].fill_between(test_dates, test_actual, test_pred,
                              color='red', alpha=0.3, label='Errore')

        if len(test_dates) > 0:
            axes[1,0].axvline(x=test_dates[0], color='red', linestyle=':',
                             linewidth=2, alpha=0.8, label='Inizio Test')

        axes[1,0].set_title('CLOSE-UP: Previsioni ARIMA vs Test Set', fontweight='bold', fontsize=12)
        axes[1,0].set_ylabel('Numero Voli')
        axes[1,0].legend(fontsize=9)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)

    else:
        if len(data) >= 12:
            try:
                decomposition = seasonal_decompose(data['Close'], model='additive', period=12, extrapolate_trend='freq')
                axes[1,0].plot(data.index, decomposition.seasonal, 'g-', linewidth=1)
                axes[1,0].set_title('Componente Stagionale (periodo 12)', fontweight='bold')
                axes[1,0].set_ylabel('Componente Stagionale')
                axes[1,0].grid(True, alpha=0.3)
            except:
                axes[1,0].text(0.5, 0.5, 'Decomposizione\nnon disponibile',
                              transform=axes[1,0].transAxes, ha='center', va='center')

    if 'test_data' in metrics and 'test_forecasts' in metrics:
        test_dates = metrics['test_dates']
        test_actual = metrics['test_data']
        test_pred = metrics['test_forecasts']

        errors = test_actual - test_pred
        error_pcts = (errors / test_actual) * 100

        colors = ['red' if e > 0 else 'blue' for e in errors]
        bars = axes[1,1].bar(range(len(errors)), errors, color=colors, alpha=0.7)

        date_labels = [d.strftime('%Y-%m') for d in test_dates]
        axes[1,1].set_xticks(range(len(errors)))
        axes[1,1].set_xticklabels(date_labels, rotation=45)

        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

        axes[1,1].set_title('Errori Mensili ARIMA (Reale - Previsto)', fontweight='bold')
        axes[1,1].set_ylabel('Errore (voli)')
        axes[1,1].grid(True, alpha=0.3)

    else:
        if len(data) >= 12:
            try:
                decomposition = seasonal_decompose(data['Close'], model='additive', period=12, extrapolate_trend='freq')
                axes[1,1].plot(data.index, decomposition.trend, 'purple', linewidth=2)
                axes[1,1].set_title('Componente Trend', fontweight='bold')
                axes[1,1].set_ylabel('Trend')
                axes[1,1].grid(True, alpha=0.3)
            except:
                axes[1,1].text(0.5, 0.5, 'Trend\nnon disponibile',
                              transform=axes[1,1].transAxes, ha='center', va='center')

    axes[1,2].axis('off')

    r2_val = metrics['r2_insample']
    mse_val = metrics['mse_insample']
    rmse_val = metrics['rmse_insample']
    mae_val = metrics['mae_insample']
    mape_val = metrics['mape_insample']
    aic_val = forecast_result['aic']
    ljung_val = metrics['ljung_val']


    if 'test_r2' in metrics:
        test_r2 = metrics['test_r2']
        test_mse = metrics['test_mse']
        test_rmse = metrics['test_rmse']
        test_mae = metrics['test_mae']
        test_mape = metrics['test_mape']
        test_size = metrics['test_size']

        metrics_text = f"""METRICHE MODELLO ARIMA
ARIMA{order}
{'='*35}

IN-SAMPLE (Training):
R² Score: {r2_val:.4f}
RMSE: {rmse_val:.0f} voli
MAE: {mae_val:.0f} voli
MAPE: {mape_val:.2f}%

OUT-OF-SAMPLE (Test):
R² Score: {test_r2:.4f}
RMSE: {test_rmse:.0f} voli
MAE: {test_mae:.0f} voli
MAPE: {test_mape:.2f}%
Test Size: {test_size} mesi

DIAGNOSTICHE:
AIC: {aic_val:.2f}
Ljung-Box p-value: {ljung_val:.4f}
        """

    axes[1,2].text(0.1, 0.9, metrics_text, transform=axes[1,2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.dropna(), ax=axes[2,0], lags=20, alpha=0.05)
    axes[2,0].set_title('ACF Residui', fontweight='bold')
    axes[2,0].grid(True, alpha=0.3)

    axes[2,1].hist(residuals.dropna(), bins=20, alpha=0.7, color='skyblue',
                   edgecolor='black', density=True)
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy import stats
    axes[2,1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
    axes[2,1].set_title('Distribuzione Residui', fontweight='bold')
    axes[2,1].set_xlabel('Residui')
    axes[2,1].set_ylabel('Densità')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)

    if 'test_data' in metrics and 'test_forecasts' in metrics:
        test_actual = metrics['test_data']
        test_pred = metrics['test_forecasts']

        cumulative_mape = []
        for i in range(1, len(test_actual) + 1):
            actual_subset = test_actual.iloc[:i]
            pred_subset = test_pred.iloc[:i]
            mape = np.mean(np.abs((actual_subset - pred_subset) / actual_subset)) * 100
            cumulative_mape.append(mape)

        months = list(range(1, len(cumulative_mape) + 1))
        axes[2,2].plot(months, cumulative_mape, 'g-', linewidth=2, marker='o')
        axes[2,2].set_title('MAPE Cumulativo ARIMA per Orizzonte', fontweight='bold')
        axes[2,2].set_xlabel('Mesi di Previsione')
        axes[2,2].set_ylabel('MAPE (%)')
        axes[2,2].grid(True, alpha=0.3)

    else:
        axes[2,2].axis('off')
        axes[2,2].text(0.5, 0.5, 'ACCURATEZZA ORIZZONTE\n(Richiede Test Set)',
                      transform=axes[2,2].transAxes, ha='center', va='center',
                      fontsize=12, style='italic')

    plt.tight_layout()

    import os
    os.makedirs('RESULTS', exist_ok=True)

    plt.savefig('RESULTS/ARIMA_European_Flights_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def save_results_to_csv(data, forecast_result, metrics, order):
    """
    Salva risultati in CSV
    """
    historical_data = pd.DataFrame({
        'date': data.index,
        'actual_value': data['Close'],
        'fitted_value': forecast_result['fitted_model'].fittedvalues,
        'residual': forecast_result['residuals'],
        'type': 'historical'
    })

    forecast_dates = forecast_result['forecast_values'].index
    forecast_data = pd.DataFrame({
        'date': forecast_dates,
        'actual_value': np.nan,
        'fitted_value': forecast_result['forecast_values'],
        'residual': np.nan,
        'type': 'forecast'
    })

    results_df = pd.concat([historical_data, forecast_data], ignore_index=True)

    conf_int = forecast_result['confidence_intervals']
    results_df.loc[results_df['type'] == 'forecast', 'conf_lower'] = conf_int.iloc[:, 0].values
    results_df.loc[results_df['type'] == 'forecast', 'conf_upper'] = conf_int.iloc[:, 1].values

    import os
    os.makedirs('RESULTS', exist_ok=True)
    results_df.to_csv('RESULTS/ARIMA_European_Flights_Results.csv', index=False)
    print(f"Risultati salvati: RESULTS/ARIMA_European_Flights_Results.csv")

def main():
    print("ARIMA SEMPLICE - VOLI EUROPEI 2016+")
    print("="*60)

    file_path = 'DATA/flights.csv'

    data = pd.read_csv(file_path)

    print(f"Colonne disponibili: {list(data.columns)}")
    print(f"Prime 3 righe:")
    print(data.head(3))

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

    series_stats = analyze_time_series(data)

    covid_start = pd.Timestamp('2020-03-01')
    has_covid = (data.index >= covid_start).any()

    if has_covid and len(data) > 50:
        split_date = pd.Timestamp('2023-01-01')
        train_data = data[data.index < split_date]
        test_data = data[data.index >= split_date]

        if len(test_data) == 0:
            train_size = len(data) - 12
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

        print(f"\nDIVISIONE DATI (COVID-aware):")

    else:
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        print(f"\nDIVISIONE DATI (80/20):")

    print(f"Training: {len(train_data)} obs ({train_data.index[0].strftime('%Y-%m-%d')} → {train_data.index[-1].strftime('%Y-%m-%d')})")
    if len(test_data) > 0:
        print(f"Test: {len(test_data)} obs ({test_data.index[0].strftime('%Y-%m-%d')} → {test_data.index[-1].strftime('%Y-%m-%d')})")

    optimal_order, best_aic, aic_results = auto_arima_order_selection(train_data['Close'])

    if len(test_data) > 0:
        forecast_periods = len(test_data)
        print(f"\nPrevisioni ARIMA per {forecast_periods} periodi del test set")
    else:
        forecast_periods = 12
        print(f"\nNessun test set disponibile - previsioni per {forecast_periods} mesi futuri")

    forecast_result = fit_arima_and_forecast(train_data['Close'], optimal_order, forecast_periods)


    fitted_values = forecast_result['fitted_model'].fittedvalues
    metrics = calculate_forecast_metrics(train_data['Close'], fitted_values)

    if len(test_data) > 0:
        forecast_values = forecast_result['forecast_values']
        n_test = min(len(test_data), len(forecast_values))

        print(f"\nCONFRONTO PREVISIONI ARIMA vs TEST SET REALE:")
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

        print(f"\nCONFRONTO DETTAGLIATO (primi 5 mesi):")
        for i in range(min(5, n_test)):
            real_val = test_data['Close'].iloc[i]
            pred_val = forecast_values.iloc[i]
            error = abs(real_val - pred_val)
            error_pct = (error / real_val) * 100
            date_str = test_data.index[i].strftime('%Y-%m')
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

    visualize_arima_results(data, forecast_result, optimal_order, metrics)

    save_results_to_csv(data, forecast_result, metrics, optimal_order)

    print(f"\n{'='*60}")
    print(f"ARIMA{optimal_order} VOLI EUROPEI COMPLETATO")
    print(f"AIC del modello: {forecast_result['aic']:.2f}")
    print(f"R² in-sample: {metrics['r2_insample']:.4f}")
    print(f"RMSE in-sample: {metrics['rmse_insample']:.0f} voli")
    if 'test_r2' in metrics:
        print(f"R² out-of-sample: {metrics['test_r2']:.4f}")
        print(f"RMSE out-of-sample: {metrics['test_rmse']:.0f} voli")
    print(f"Grafici salvati: RESULTS/ARIMA_European_Flights_Analysis.png")
    print(f"Dati salvati: RESULTS/ARIMA_European_Flights_Results.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()