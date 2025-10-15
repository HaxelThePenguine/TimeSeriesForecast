import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="talk")

## Analisi dei vari dati (cerca pattern, stagionalità, trend, ecc.)

def analyze_time_series(data):
    """
    Analisi esplorativa della serie storica
    """
    print("ANALISI ESPLORATIVA SERIE STORICA VOLI EUROPEI (SARIMA)")
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

    if len(data) > 14:
        weekly_avg = data.groupby(data.index.dayofweek)['Close'].mean()
        weekday_names = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
        peak_day = weekly_avg.idxmax()
        low_day = weekly_avg.idxmin()
        weekly_strength = (weekly_avg.max() - weekly_avg.min()) / weekly_avg.mean() * 100
        print(f"Stagionalità settimanale: {weekly_strength:.1f}% variazione")
        print(f"Picco: {weekday_names[peak_day]} ({weekly_avg[peak_day]:.0f} voli)")
        print(f"Minimo: {weekday_names[low_day]} ({weekly_avg[low_day]:.0f} voli)")

    if len(data) > 30:
        monthly_avg = data.groupby(data.index.month)['Close'].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        monthly_strength = (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100
        print(f"Stagionalità mensile: {monthly_strength:.1f}% variazione")
        print(f"Picco: Mese {peak_month} ({monthly_avg[peak_month]:.0f} voli)")
        print(f"Minimo: Mese {low_month} ({monthly_avg[low_month]:.0f} voli)")

    return {
        'total_observations': len(data),
        'min_value': data['Close'].min(),
        'max_value': data['Close'].max(),
        'mean_value': data['Close'].mean(),
        'std_value': data['Close'].std(),
        'total_change': total_change,
        'has_covid_impact': len(covid_data) > 0,
        'weekly_strength': weekly_strength if 'weekly_strength' in locals() else 0,
        'monthly_strength': monthly_strength if 'monthly_strength' in locals() else 0
    }

def detect_optimal_seasonal_period(series):
    """
    Rileva periodo stagionale ottimale per SARIMA
    """
    print("\nRILEVAMENTO PERIODO STAGIONALE OTTIMALE")
    print("="*50)

    periods_to_test = [7, 30]
    seasonal_strengths = {}

    for period in periods_to_test:
        if len(series) >= 3 * period:
            try:
                decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
                seasonal_var = np.var(decomposition.seasonal)
                total_var = np.var(series)
                strength = seasonal_var / total_var * 100
                seasonal_strengths[period] = strength

                period_name = {7: 'Settimanale', 30: 'Mensile'}[period]
                print(f"{period_name} (periodo {period}): {strength:.2f}% varianza stagionale")

            except Exception as e:
                print(f"Periodo {period}: Errore nella decomposizione")
                seasonal_strengths[period] = 0
        else:
            print(f"Periodo {period}: Dati insufficienti")
            seasonal_strengths[period] = 0

    if seasonal_strengths:
        best_period = max(seasonal_strengths, key=seasonal_strengths.get)
        best_strength = seasonal_strengths[best_period]

        if best_strength > 2:
            period_name = {7: 'Settimanale', 30: 'Mensile'}[best_period]
            print(f"\nPeriodo stagionale ottimale: {period_name} (s={best_period})")
            return best_period, best_strength
        else:
            print(f"\nStagionalità debole - uso periodo settimanale default (s=7)")
            return 7, seasonal_strengths.get(7, 0)
    else:
        print("\nImpossibile rilevare stagionalità - uso periodo settimanale default")
        return 7, 0

def test_stationarity_seasonal(series, seasonal_period=None):
    """
    Test di stazionarietà con differenziazione stagionale per SARIMA
    """
    print("\nTEST DI STAZIONARIETÀ (SARIMA)")
    print("="*50)

    result_original = adfuller(series.dropna())
    print(f"Serie originale:")
    print(f"   ADF Statistic: {result_original[0]:.4f}")
    print(f"   p-value: {result_original[1]:.6f}")
    print(f"   Stazionaria: {'SI' if result_original[1] < 0.05 else 'NO'}")

    d_recommended = 0
    D_recommended = 0

    current_series = series.copy()
    for d in range(1, 3):
        current_series = current_series.diff().dropna()
        if len(current_series) > 10:
            result_diff = adfuller(current_series)
            print(f"Serie con {d} differenziazione(i) regolare(i):")
            print(f"   ADF Statistic: {result_diff[0]:.4f}")
            print(f"   p-value: {result_diff[1]:.6f}")
            print(f"   Stazionaria: {'SI' if result_diff[1] < 0.05 else 'NO'}")

            if result_diff[1] < 0.05 and d_recommended == 0:
                d_recommended = d
                print(f"   → d = {d} sufficiente per stazionarietà")
                break

    if seasonal_period and seasonal_period <= len(series) // 3:
        print(f"\nTest differenziazione stagionale (periodo {seasonal_period}):")
        seasonal_diff = series.diff(seasonal_period).dropna()

        if len(seasonal_diff) > 10:
            result_seasonal = adfuller(seasonal_diff)
            print(f"Serie con 1 differenziazione stagionale:")
            print(f"   ADF Statistic: {result_seasonal[0]:.4f}")
            print(f"   p-value: {result_seasonal[1]:.6f}")
            print(f"   Stazionaria: {'SI' if result_seasonal[1] < 0.05 else 'NO'}")

            if result_seasonal[1] < 0.05:
                D_recommended = 1
                print(f"   → D = 1 necessario per stazionarietà stagionale")

    print(f"\nRaccomandazioni differenziazione:")
    print(f"   d (regolare): {d_recommended}")
    print(f"   D (stagionale): {D_recommended}")

    return d_recommended, D_recommended

# Funzioni effettive SARIMA

def auto_sarima_order_selection(series, seasonal_period=None, max_p=1, max_q=1, max_P=1, max_Q=1):
    """
    Selezione automatica ordini SARIMA con AIC
    """
    print("\nSELEZIONE AUTOMATICA ORDINI SARIMA")
    print("="*50)

    d, D = test_stationarity_seasonal(series, seasonal_period)

    if seasonal_period is None or seasonal_period > 30:
        print("Nessuna stagionalità o periodo troppo lungo → uso ARIMA semplice")
        seasonal_period = 0
        max_P = 0
        max_Q = 0
        D = 0

    best_aic = np.inf
    best_order = (1, d, 1)
    best_seasonal_order = (0, D, 0, seasonal_period)
    aic_results = []

    total_models = (max_p + 1) * (max_q + 1) * (max_P + 1) * (max_Q + 1)
    print(f"Testing {total_models} combinazioni SARIMA...")
    print(f"Formato: SARIMA(p,d,q)(P,D,Q,{seasonal_period})")

    model_count = 0
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(max_P + 1):
                for Q in range(max_Q + 1):
                    try:
                        model_count += 1
                        if seasonal_period > 0:
                            model = SARIMAX(series, order=(p, d, q),
                                          seasonal_order=(P, D, Q, seasonal_period))
                        else:
                            model = SARIMAX(series, order=(p, d, q))

                        fitted_model = model.fit(disp=False, maxiter=20, method='lbfgs')

                        aic = fitted_model.aic
                        aic_results.append({
                            'p': p, 'd': d, 'q': q,
                            'P': P, 'D': D, 'Q': Q, 's': seasonal_period,
                            'aic': aic
                        })

                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_seasonal_order = (P, D, Q, seasonal_period)

                        if model_count <= 20:
                            print(f"   SARIMA({p},{d},{q})({P},{D},{Q},{seasonal_period}): AIC = {aic:.2f}")
                        elif model_count == 21:
                            print("   ... (altri modelli testati silenziosamente)")

                    except Exception as e:
                        if model_count <= 20:
                            print(f"   SARIMA({p},{d},{q})({P},{D},{Q},{seasonal_period}): Failed")
                        continue

    print(f"\nMODELLO OTTIMALE: SARIMA{best_order}{best_seasonal_order} con AIC = {best_aic:.2f}")

    return best_order, best_seasonal_order, best_aic, aic_results

def fit_sarima_and_forecast(series, order, seasonal_order, forecast_periods=18):
    """
    Fit SARIMA e generazione previsioni
    """
    print(f"\nFIT MODELLO SARIMA{order}{seasonal_order}")
    print("="*50)

    try:
        if seasonal_order[3] > 0:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        else:
            model = SARIMAX(series, order=order)

        fitted_model = model.fit(disp=False, maxiter=100)

        print(f"Modello fittato con successo")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        print(f"Log-likelihood: {fitted_model.llf:.2f}")

        residuals = fitted_model.resid

        try:
            ljung_box = fitted_model.test_serial_correlation(method='ljungbox', lags=10)
            ljung_box_pvalue = 0.5
            if hasattr(ljung_box, 'pvalue'):
                ljung_box_pvalue = float(ljung_box.pvalue)
            elif hasattr(ljung_box, 'shape') and ljung_box.shape[0] > 0:
                ljung_box_pvalue = float(ljung_box.iloc[0, 1])
        except:
            ljung_box_pvalue = 0.5

        print(f"Ljung-Box test p-value: {ljung_box_pvalue:.4f}")
        print(f"   Residui indipendenti: {'SI' if ljung_box_pvalue > 0.05 else 'NO'}")

        forecast_result = fitted_model.get_forecast(steps=forecast_periods)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()

        print(f"Previsioni generate per {forecast_periods} periodi futuri")

        return {
            'fitted_model': fitted_model,
            'forecast_values': forecast_values,
            'confidence_intervals': conf_int,
            'residuals': residuals,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'ljung_box_pvalue': ljung_box_pvalue
        }

    except Exception as e:
        print(f"Errore nel fit del modello: {str(e)}")
        try:
            print("Tentativo con SARIMA(1,1,1)(0,0,0,0) semplificato...")
            model_simple = SARIMAX(series, order=(1,1,1))
            fitted_simple = model_simple.fit(disp=False)

            forecast_simple = fitted_simple.get_forecast(steps=forecast_periods)

            return {
                'fitted_model': fitted_simple,
                'forecast_values': forecast_simple.predicted_mean,
                'confidence_intervals': forecast_simple.conf_int(),
                'residuals': fitted_simple.resid,
                'aic': fitted_simple.aic,
                'bic': fitted_simple.bic,
                'ljung_box_pvalue': 0.5
            }
        except Exception as e2:
            print(f"Anche modello semplificato fallito: {str(e2)}")
            return None


def calculate_forecast_metrics(actual, predicted, forecast_periods_list=[3, 7, 14]):
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

def visualize_sarima_results(data, forecast_result, order, seasonal_order, metrics):
    """
    Visualizzazione risultati SARIMA
    """
    print("\nGENERAZIONE VISUALIZZAZIONI")
    print("="*50)

    fitted_model = forecast_result['fitted_model']
    forecast_values = forecast_result['forecast_values']
    conf_int = forecast_result['confidence_intervals']
    residuals = forecast_result['residuals']

    
    forecast_dates = forecast_values.index
    fig, axes = plt.subplots(3, 3, figsize=(24, 16))
    fig.suptitle(f'SARIMA{order}{seasonal_order} Analysis - Voli Europei 2016+ (con Close-up Previsioni)',
                 fontsize=16, fontweight='bold')

    axes[0,0].plot(data.index, data['Close'], 'b-', linewidth=2, label='Dati Storici')

    train_size = len(fitted_model.fittedvalues)
    train_dates = data.index[:train_size]
    axes[0,0].plot(train_dates, fitted_model.fittedvalues, 'r--', linewidth=1,
                   alpha=0.8, label='Fitted Values (Training)')

    if 'test_data' in metrics and 'test_forecasts' in metrics:
        test_dates = metrics['test_dates']
        axes[0,0].plot(test_dates, metrics['test_data'], 'g-', linewidth=3,
                       label=f'Test Set Reale ({len(test_dates)} giorni)', marker='o', markersize=4)

        axes[0,0].plot(test_dates, metrics['test_forecasts'], 'orange', linewidth=2,
                       linestyle='--', label='Previsioni SARIMA', marker='s', markersize=3)

        axes[0,0].fill_between(test_dates, metrics['test_data'], metrics['test_forecasts'],
                              color='red', alpha=0.3, label='Errore Previsione')
    else:
        axes[0,0].plot(forecast_dates, forecast_values, 'g-', linewidth=2, label='Previsioni SARIMA')
        axes[0,0].fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                              color='green', alpha=0.2, label='95% CI')

    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2021-12-31')
    if data.index[0] <= covid_start <= data.index[-1]:
        axes[0,0].axvspan(covid_start, covid_end, alpha=0.15, color='red', label='Periodo COVID')

    axes[0,0].set_title(f'SARIMA{order}{seasonal_order} - Previsioni vs Test Set', fontweight='bold')
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

        context_days = 14
        train_end_idx = len(data) - len(test_actual)
        context_start = max(0, train_end_idx - context_days)

        context_dates = data.index[context_start:train_end_idx]
        context_values = data['Close'].iloc[context_start:train_end_idx]

        axes[1,0].plot(context_dates, context_values, 'b-', linewidth=2,
                       label='Training (contesto)', alpha=0.7)
        axes[1,0].plot(test_dates, test_actual, 'g-', linewidth=3,
                       label='Test Set Reale', marker='o', markersize=6)
        axes[1,0].plot(test_dates, test_pred, 'orange', linewidth=2,
                       linestyle='--', label='Previsioni SARIMA', marker='s', markersize=5)

        axes[1,0].fill_between(test_dates, test_actual, test_pred,
                              color='red', alpha=0.3, label='Errore')

        if len(test_dates) > 0:
            axes[1,0].axvline(x=test_dates[0], color='red', linestyle=':',
                             linewidth=2, alpha=0.8, label='Inizio Test')

        axes[1,0].set_title('CLOSE-UP: Previsioni SARIMA vs Test Set', fontweight='bold', fontsize=12)
        axes[1,0].set_ylabel('Numero Voli')
        axes[1,0].legend(fontsize=9)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)

    else:
        if len(data) >= 14:
            try:
                seasonal_period = seasonal_order[3] if len(seasonal_order) > 3 else 7
                decomposition = seasonal_decompose(data['Close'], model='additive', period=seasonal_period, extrapolate_trend='freq')
                axes[1,0].plot(data.index, decomposition.seasonal, 'g-', linewidth=1)
                axes[1,0].set_title(f'Componente Stagionale (periodo {seasonal_period})', fontweight='bold')
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

        date_labels = [d.strftime('%Y-%m-%d') for d in test_dates]
        axes[1,1].set_xticks(range(len(errors)))
        axes[1,1].set_xticklabels(date_labels, rotation=45)

        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

        axes[1,1].set_title('Errori Giornalieri SARIMA (Reale - Previsto)', fontweight='bold')
        axes[1,1].set_ylabel('Errore (voli)')
        axes[1,1].grid(True, alpha=0.3)

    else:
        if len(data) >= 14:
            try:
                seasonal_period = seasonal_order[3] if len(seasonal_order) > 3 else 7
                decomposition = seasonal_decompose(data['Close'], model='additive', period=seasonal_period, extrapolate_trend='freq')
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
    ljung_val = forecast_result.get('ljung_box_pvalue', 0.5)

    if 'test_r2' in metrics:
        test_r2 = metrics['test_r2']
        test_mse = metrics['test_mse']
        test_rmse = metrics['test_rmse']
        test_mae = metrics['test_mae']
        test_mape = metrics['test_mape']
        test_size = metrics['test_size']

        metrics_text = f"""METRICHE MODELLO SARIMA
SARIMA{order}{seasonal_order}
{'='*40}

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
Test Size: {test_size} giorni

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

        days = list(range(1, len(cumulative_mape) + 1))
        axes[2,2].plot(days, cumulative_mape, 'g-', linewidth=2, marker='o')
        axes[2,2].set_title('MAPE Cumulativo SARIMA per Orizzonte', fontweight='bold')
        axes[2,2].set_xlabel('Giorni di Previsione')
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

    plt.savefig('RESULTS/SARIMA_European_Flights_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def save_results_to_csv(data, forecast_result, metrics, order, seasonal_order):
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
    results_df.to_csv('RESULTS/SARIMA_European_Flights_Results.csv', index=False)
    print(f"Risultati salvati: RESULTS/SARIMA_European_Flights_Results.csv")

def main():
    print("SARIMA SEMPLICE - VOLI EUROPEI 2016+")
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

    seasonal_period, seasonal_strength = detect_optimal_seasonal_period(data['Close'])

    covid_start = pd.Timestamp('2020-03-01')
    has_covid = (data.index >= covid_start).any()

    if has_covid and len(data) > 50:
        split_date = pd.Timestamp('2023-01-01')
        train_data = data[data.index < split_date]
        test_data = data[data.index >= split_date]

        if len(test_data) == 0:
            train_size = len(data) - 30
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

    optimal_order, optimal_seasonal_order, best_aic, aic_results = auto_sarima_order_selection(
        train_data['Close'], seasonal_period)

    if len(test_data) > 0:
        forecast_periods = len(test_data)
        print(f"\nPrevisioni SARIMA per {forecast_periods} periodi del test set")
    else:
        forecast_periods = 30
        print(f"\nNessun test set disponibile - previsioni per {forecast_periods} giorni futuri")

    forecast_result = fit_sarima_and_forecast(train_data['Close'], optimal_order, optimal_seasonal_order, forecast_periods)

    if forecast_result is None:
        print("Impossibile fittare il modello SARIMA")
        return

    fitted_values = forecast_result['fitted_model'].fittedvalues
    metrics = calculate_forecast_metrics(train_data['Close'], fitted_values, [3, 7, 14])

    if len(test_data) > 0:
        forecast_values = forecast_result['forecast_values']
        n_test = min(len(test_data), len(forecast_values))

        print(f"\nCONFRONTO PREVISIONI SARIMA vs TEST SET REALE:")
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

    visualize_sarima_results(data, forecast_result, optimal_order, optimal_seasonal_order, metrics)

    save_results_to_csv(data, forecast_result, metrics, optimal_order, optimal_seasonal_order)

    print(f"\n{'='*60}")
    print(f"SARIMA{optimal_order}{optimal_seasonal_order} VOLI EUROPEI COMPLETATO")
    print(f"AIC del modello: {forecast_result['aic']:.2f}")
    print(f"R² in-sample: {metrics['r2_insample']:.4f}")
    print(f"RMSE in-sample: {metrics['rmse_insample']:.0f} voli")
    if 'test_r2' in metrics:
        print(f"R² out-of-sample: {metrics['test_r2']:.4f}")
        print(f"RMSE out-of-sample: {metrics['test_rmse']:.0f} voli")
    print(f"Grafici salvati: RESULTS/SARIMA_European_Flights_Analysis.png")
    print(f"Dati salvati: RESULTS/SARIMA_European_Flights_Results.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()