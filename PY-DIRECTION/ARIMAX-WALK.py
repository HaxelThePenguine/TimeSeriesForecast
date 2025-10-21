import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="talk")


## qui tutto uguale ad ARIMA-WALK.py ma con variabili esogene (ARIMAX) stessa roba



## HELPER FUNCTIONS
## possibile implementare anche qua la selezione automatica delle exog con metodi più efficienti e utili 
## ma per ora va bene così

def check_stationarity(series, alpha=0.05):
    """
    Test di stazionarietà ADF (Augmented Dickey-Fuller)
    """
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return result[1] < alpha, result[1]
    except:
        return False, 1.0

### PREPARAZIONE DATI E SELEZIONE VARIABILI
def prepare_exogenous_variables(data, vix_data, us10y_data, gold_data, oil_data):
    """
    Prepara e allinea tutte le variabili esogene per ARIMAX
    """
    print("Preparazione variabili esogene per ARIMAX...")

    combined = data[['Close']].copy()
    combined.columns = ['SP500']

    if 'Close' in vix_data.columns:
        combined = combined.merge(vix_data[['Close']].rename(columns={'Close': 'VIX'}),
                                left_index=True, right_index=True, how='left')

    if 'Close' in us10y_data.columns:
        combined = combined.merge(us10y_data[['Close']].rename(columns={'Close': 'US10Y'}),
                                left_index=True, right_index=True, how='left')

    if 'Close' in gold_data.columns:
        combined = combined.merge(gold_data[['Close']].rename(columns={'Close': 'GOLD'}),
                                left_index=True, right_index=True, how='left')

    if 'Close' in oil_data.columns:
        combined = combined.merge(oil_data[['Close']].rename(columns={'Close': 'OIL'}),
                                left_index=True, right_index=True, how='left')

    combined = combined.fillna(method='ffill').dropna()

    print(f"Dataset combinato: {len(combined)} osservazioni")
    print(f"Variabili disponibili: {list(combined.columns)}")
    print(f"Periodo: {combined.index[0].date()} → {combined.index[-1].date()}")

    return combined

def select_significant_exog_variables(endog, exog_candidates, max_vars=3, significance_level=0.05):
    """
    Selezione automatica delle variabili esogene più significative
    """

    ## qui usamo correlazione semplice, si può migliorare con metodi più sofisticati
    try:
        if len(exog_candidates.columns) == 0:
            return pd.DataFrame()

        correlations = {}
        endog_returns = endog.pct_change().dropna()

        for col in exog_candidates.columns:
            try:
                exog_aligned = exog_candidates[col].reindex(endog_returns.index).fillna(method='ffill')
                if len(exog_aligned) > 10:
                    correlation = abs(np.corrcoef(endog_returns, exog_aligned.pct_change().fillna(0))[0,1])
                    if not np.isnan(correlation):
                        correlations[col] = correlation
            except:
                continue

        if len(correlations) == 0:
            return pd.DataFrame()

        sorted_vars = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_vars = [var[0] for var in sorted_vars[:max_vars]]

        selected_exog = exog_candidates[selected_vars].copy()

        print(f"Variabili esogene selezionate: {selected_vars}")
        print(f"Correlazioni: {[f'{var}: {correlations[var]:.3f}' for var in selected_vars]}")

        return selected_exog

    except Exception as e:
        print(f"Errore selezione variabili esogene: {e}")
        return pd.DataFrame()

def auto_arimax_order_selection(endog, exog, max_p=2, max_q=2):
    """
    Selezione automatica ordine ARIMAX con AIC
    """
    try:
        is_stationary, p_value = check_stationarity(endog)
        d = 0 if is_stationary else 1

        candidate_orders = [
            (1, d, 1), (0, d, 1), (1, d, 0),
            (2, d, 1), (1, d, 2), (0, d, 0)
        ]

        best_aic = np.inf
        best_order = (1, d, 1)

        for order in candidate_orders:
            try:
                if len(exog.columns) > 0:
                    model = ARIMA(endog, order=order, exog=exog)
                else:
                    model = ARIMA(endog, order=order)

                fitted_model = model.fit(
                    method_kwargs={'warn_convergence': False, 'maxiter': 15},
                    low_memory=True
                )

                aic = fitted_model.aic
                if not np.isnan(aic) and aic < best_aic:
                    best_aic = aic
                    best_order = order

            except:
                continue

        return best_order, best_aic

    except:
        return (1, 1, 1), 1000



## FIT E PREDIZIONE

def fit_arimax_predict(endog, exog, order=(1,1,1), steps=1, exog_forecast=None):
    """
    Fit ARIMAX e predizione con variabili esogene
    """
    try:
        if len(endog) < 30:
            return endog.iloc[-1] * 1.001, 0.5, 1000

        if len(exog.columns) > 0 and len(exog) > 0:
            min_len = min(len(endog), len(exog)) # ocio errori
            endog_aligned = endog.iloc[-min_len:]
            exog_aligned = exog.iloc[-min_len:]

            # stesso metodo di arima ma con exog
            model = ARIMA(endog_aligned, order=order, exog=exog_aligned)
        else:
            model = ARIMA(endog, order=order)

        fitted_model = model.fit(
            method_kwargs={'warn_convergence': False, 'maxiter': 15},
            low_memory=True
        )

        if len(exog.columns) > 0 and exog_forecast is not None:
            forecast = fitted_model.forecast(steps=steps, exog=exog_forecast)
        else:
            forecast = fitted_model.forecast(steps=steps)

        predicted_price = forecast[0] if hasattr(forecast, '__len__') else forecast
        confidence = 0.65

        return predicted_price, confidence, fitted_model.aic

    except Exception as e:
        try:
            recent_returns = endog.pct_change().tail(10).mean()
            predicted_price = endog.iloc[-1] * (1 + recent_returns)
            return predicted_price, 0.4, 1000
        except:
            return endog.iloc[-1], 0.3, 1000

def train_arimax_predict_direction(combined_data, target_col='SP500', auto_order=True):
    """
    Addestra ARIMAX e predice direzione con variabili esogene
    """
    try:
        if len(combined_data) < 50:
            return {
                'predicted_direction': 1,
                'expected_return': 0.001,
                'confidence': 0.5,
                'arimax_order': (1,1,1),
                'aic': 1000,
                'selected_exog_vars': []
            }

        endog = combined_data[target_col]
        exog_candidates = combined_data.drop(columns=[target_col])

        selected_exog = select_significant_exog_variables(endog, exog_candidates, max_vars=3)

        if auto_order:
            order, aic = auto_arimax_order_selection(endog, selected_exog)
        else:
            order = (1, 1, 1)
            aic = 1000

        if len(selected_exog.columns) > 0:
            exog_forecast = selected_exog.iloc[-1:].values.reshape(1, -1)
        else:
            exog_forecast = None

        # Fit ARIMAX e predizione
        predicted_price, confidence, final_aic = fit_arimax_predict(
            endog, selected_exog, order, steps=1, exog_forecast=exog_forecast
        )

        current_price = endog.iloc[-1]
        expected_return = (predicted_price - current_price) / current_price
        predicted_direction = 1 if expected_return > 0 else 0

        is_stationary, stationarity_p = check_stationarity(endog)

        print(f"ARIMAX{order}, AIC: {final_aic:.2f}")
        print(f"Prezzo: {current_price:.2f} → {predicted_price:.2f}")
        print(f"Confidence: {confidence:.3f}, Exog vars: {len(selected_exog.columns)}")

        return {
            'predicted_direction': predicted_direction,
            'expected_return': expected_return,
            'confidence': confidence,
            'arimax_order': order,
            'aic': final_aic,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'is_stationary': is_stationary,
            'stationarity_pvalue': stationarity_p,
            'selected_exog_vars': list(selected_exog.columns) if len(selected_exog.columns) > 0 else []
        }

    except Exception as e:
        print(f"Errore ARIMAX: {e}")
        avg_return = combined_data[target_col].pct_change().mean() if len(combined_data) > 1 else 0.001
        return {
            'predicted_direction': 1 if avg_return > 0 else 0,
            'expected_return': avg_return,
            'confidence': 0.5,
            'arimax_order': (1,1,1),
            'aic': 1000,
            'selected_exog_vars': []
        }

def walk_forward_arimax_analysis(combined_data, window_size, target_col='SP500', auto_order=True):
    """
    Walk-forward ARIMAX analysis con variabili esogene
    """
    mode_text = "AUTO-ARIMAX" if auto_order else "ARIMAX(1,1,1)"
    print(f"WALK-FORWARD {mode_text} ANALYSIS...")
    print(f"Periodo analisi: {combined_data.index[window_size].date()} → {combined_data.index[-1].date()}")
    print(f"Numero predizioni: {len(combined_data) - window_size}")
    print(f"Variabili esogene disponibili: {list(combined_data.columns)}")
    print("="*60)

    all_predictions = []
    start_time = time.time()

    for i in range(window_size, len(combined_data)):
        window_data = combined_data.iloc[i-window_size:i]
        current_date = combined_data.index[i]

        print(f"\nPredizione #{i-window_size+1}/{len(combined_data)-window_size}")
        print(f"Training Window: {window_data.index[0].strftime('%Y-%m-%d')} → {window_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Predicting: {current_date.strftime('%Y-%m-%d')}")

        prediction = train_arimax_predict_direction(window_data, target_col, auto_order)

        actual_price = combined_data.iloc[i][target_col]
        previous_price = combined_data.iloc[i-1][target_col]
        actual_return = (actual_price - previous_price) / previous_price
        actual_direction = 1 if actual_return > 0 else 0

        result = {
            'date': current_date,
            'predicted_direction': prediction['predicted_direction'],
            'actual_direction': actual_direction,
            'predicted_return': prediction['expected_return'],
            'actual_return': actual_return,
            'confidence': prediction['confidence'],
            'arima_order': prediction['arimax_order'],
            'aic': prediction['aic'],
            'predicted_price': prediction.get('predicted_price', previous_price),
            'actual_price': actual_price,
            'previous_price': previous_price,
            'is_stationary': prediction.get('is_stationary', False),
            'stationarity_pvalue': prediction.get('stationarity_pvalue', 1.0),
            'window_start': window_data.index[0],
            'window_end': window_data.index[-1],
            'selected_exog_vars': prediction.get('selected_exog_vars', []),
            'model_type': 'ARIMAX'
        }

        all_predictions.append(result)

        exog_vars_str = ', '.join(prediction.get('selected_exog_vars', []))
        print(f"ARIMAX{prediction['arimax_order']}, AIC: {prediction['aic']:.2f}")
        print(f"Exog vars: [{exog_vars_str}]")
        print(f"Confidence: {prediction['confidence']:.3f}")
        print(f"Rendimento atteso: {prediction['expected_return']:.4f}")
        print(f"Direzione predetta: {'UP' if prediction['predicted_direction'] == 1 else 'DOWN'}")
        print(f"Direzione reale: {'UP' if actual_direction == 1 else 'DOWN'}")
        print(f"Corretto: {'SI' if prediction['predicted_direction'] == actual_direction else 'NO'}")

        if (i - window_size + 1) % 100 == 0:
            recent_results = all_predictions[-min(100, len(all_predictions)):]
            recent_accuracy = np.mean([r['predicted_direction'] == r['actual_direction'] for r in recent_results]) * 100
            avg_aic = np.mean([r['aic'] for r in recent_results])
            avg_confidence = np.mean([r['confidence'] for r in recent_results])
            elapsed = time.time() - start_time
            print(f"Accuracy ultime {len(recent_results)} predizioni: {recent_accuracy:.2f}%")
            print(f"AIC medio: {avg_aic:.2f}, Confidence media: {avg_confidence:.3f}")
            print(f"Velocità: {elapsed/(i-window_size+1):.3f}s per predizione")

    return all_predictions


## METRICHE E VISUALIZZAZIONE

def calculate_metrics_and_visualize_walkforward(all_predictions):
    """
    Calcola metriche e crea visualizzazioni per risultati walk-forward ARIMAX
    """
    print("\n" + "="*60)
    print("RISULTATI FINALI ARIMAX WALK-FORWARD DIRECTIONAL PREDICTION")
    print("="*60)

    if len(all_predictions) == 0:
        print("Nessuna predizione disponibile")
        return

    results_df = pd.DataFrame(all_predictions)

    total_predictions = len(results_df)
    correct_predictions = (results_df['predicted_direction'] == results_df['actual_direction']).sum()
    directional_accuracy = (correct_predictions / total_predictions) * 100

    up_predictions = results_df[results_df['predicted_direction'] == 1]
    down_predictions = results_df[results_df['predicted_direction'] == 0]

    up_correct = (up_predictions['predicted_direction'] == up_predictions['actual_direction']).sum()
    down_correct = (down_predictions['predicted_direction'] == down_predictions['actual_direction']).sum()

    up_accuracy = (up_correct / len(up_predictions) * 100) if len(up_predictions) > 0 else 0
    down_accuracy = (down_correct / len(down_predictions) * 100) if len(down_predictions) > 0 else 0

    avg_confidence = results_df['confidence'].mean()
    avg_aic = results_df['aic'].mean()
    stationary_pct = (results_df['is_stationary']).mean() * 100

    order_counts = results_df['arima_order'].value_counts()

    exog_counts = {}
    for exog_list in results_df['selected_exog_vars']:
        for var in exog_list:
            exog_counts[var] = exog_counts.get(var, 0) + 1

    print(f"\nPERFORMANCE COMPLESSIVA:")
    print(f"Periodo analizzato: {results_df['date'].min().date()} → {results_df['date'].max().date()}")
    print(f"Totale predizioni: {total_predictions}")
    print(f"Predizioni corrette: {correct_predictions}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")

    print(f"\nBREAKDOWN PER DIREZIONE:")
    print(f"Predizioni UP: {len(up_predictions)} (Accuracy: {up_accuracy:.2f}%)")
    print(f"Predizioni DOWN: {len(down_predictions)} (Accuracy: {down_accuracy:.2f}%)")

    print(f"\nANALISI MODELLI ARIMAX:")
    print(f"Confidence media: {avg_confidence:.3f}")
    print(f"AIC medio: {avg_aic:.2f}")
    print(f"Serie stazionarie: {stationary_pct:.1f}%")
    print(f"Ordini ARIMAX più usati:")
    for order, count in order_counts.head(3).items():
        print(f"  {order}: {count} volte ({count/total_predictions*100:.1f}%)")

    print(f"\nVARIABILI ESOGENE PIÙ UTILIZZATE:")
    sorted_exog = sorted(exog_counts.items(), key=lambda x: x[1], reverse=True)
    for var, count in sorted_exog:
        print(f"  {var}: {count} volte ({count/total_predictions*100:.1f}%)")

    high_conf_mask = results_df['confidence'] > 0.7
    high_conf_accuracy = (results_df[high_conf_mask]['predicted_direction'] == results_df[high_conf_mask]['actual_direction']).mean() * 100 if high_conf_mask.sum() > 0 else 0
    print(f"Predizioni alta confidence (>0.7): {high_conf_mask.sum()}")
    print(f"Accuracy alta confidence: {high_conf_accuracy:.2f}%")

    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values('date')
    results_df['cumulative_accuracy'] = (results_df['predicted_direction'] == results_df['actual_direction']).cumsum() / (np.arange(len(results_df)) + 1) * 100

    results_df['year'] = results_df['date'].dt.year
    yearly_performance = results_df.groupby('year').apply(
        lambda x: (x['predicted_direction'] == x['actual_direction']).mean() * 100
    )

    print(f"\nPERFORMANCE ANNUALE:")
    for year, acc in yearly_performance.items():
        year_count = len(results_df[results_df['year'] == year])
        print(f"  {year}: {acc:.2f}% ({year_count} predizioni)")


    # no visualizzazioni per ora
    #create_walkforward_visualizations(results_df)

    #save_walkforward_results(results_df)


##      MAIN

def main():
    print("CONFIGURAZIONE ARIMAX WALK-FORWARD ANALYSIS")
    print("="*60)

    data_SP500 = pd.read_csv('DATA/SP500Data.csv')
    data_SP500['Date'] = pd.to_datetime(data_SP500['Date'])
    data_SP500.set_index('Date', inplace=True)
    data_SP500 = data_SP500[data_SP500.index.dayofweek < 5]

    data_SP500 = data_SP500[data_SP500.index >= '2019-01-01']

    # Caricamento dati esogeni
     # se non trovati, creare proxy semplici
    try:
        data_VIX = pd.read_csv('DATA/VIXData.csv')
        data_VIX['Date'] = pd.to_datetime(data_VIX['Date'])
        data_VIX.set_index('Date', inplace=True)
        data_VIX = data_VIX[data_VIX.index.dayofweek < 5]
    except:
        print("VIX data non trovata, creando proxy...")
        data_VIX = data_SP500[['Close']].copy()
        data_VIX['Close'] = data_SP500['Close'].pct_change().rolling(20).std() * 100

    try:
        data_US10Y = pd.read_csv('DATA/US10YData.csv')
        data_US10Y['Date'] = pd.to_datetime(data_US10Y['Date'])
        data_US10Y.set_index('Date', inplace=True)
        data_US10Y = data_US10Y[data_US10Y.index.dayofweek < 5]
    except:
        print("US10Y data non trovata, usando proxy...")
        data_US10Y = data_SP500[['Close']].copy()
        data_US10Y['Close'] = 2.5

    try:
        data_GOLD = pd.read_csv('DATA/GoldData.csv')
        data_GOLD['Date'] = pd.to_datetime(data_GOLD['Date'])
        data_GOLD.set_index('Date', inplace=True)
        data_GOLD = data_GOLD[data_GOLD.index.dayofweek < 5]
    except:
        print("Gold data non trovata, usando proxy...")
        data_GOLD = data_SP500[['Close']].copy()
        data_GOLD['Close'] = data_SP500['Close'] * 0.5

    try:
        data_OIL = pd.read_csv('DATA/OilData.csv')
        data_OIL['Date'] = pd.to_datetime(data_OIL['Date'])
        data_OIL.set_index('Date', inplace=True)
        data_OIL = data_OIL[data_OIL.index.dayofweek < 5]
    except:
        print("Oil data non trovata, usando proxy...")
        data_OIL = data_SP500[['Close']].copy()
        data_OIL['Close'] = data_SP500['Close'] * 0.3

    combined_data = prepare_exogenous_variables(data_SP500, data_VIX, data_US10Y, data_GOLD, data_OIL)

    window_size_years = 1.5
    window_size_days = int(window_size_years * 252)
    auto_order = True

    print(f"Dataset Combinato: {len(combined_data)} osservazioni dal {combined_data.index[0].date()} al {combined_data.index[-1].date()}")
    print(f"Window Size: {window_size_days} giorni (~{window_size_years} anni)")
    print(f"Predizioni Walk-Forward: {len(combined_data) - window_size_days} osservazioni")
    print(f"ARIMAX Mode: {'Auto-selection' if auto_order else 'Fixed (1,1,1)'}")
    print(f"Variabili esogene: {list(combined_data.columns)}")

    walk_forward_results = walk_forward_arimax_analysis(combined_data, window_size_days, 'SP500', auto_order)

    calculate_metrics_and_visualize_walkforward(walk_forward_results)

if __name__ == "__main__":
    main()
    