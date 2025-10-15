import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="talk")

## HELPER FUNCTIONS

def check_stationarity(series, alpha=0.05):
    """
    Test di stazionarietà con ADF test
    """
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < alpha
        return is_stationary, p_value
    except:
        return False, 1.0

def make_stationary(series, max_diff=2):
    """
    Rende la serie stazionaria tramite differenziazione
    """
    original_series = series.copy()
    diff_order = 0

    for d in range(max_diff + 1):
        if d == 0:
            test_series = series
        else:
            test_series = series.diff(d).dropna()

        is_stationary, p_value = check_stationarity(test_series)

        if is_stationary:
            diff_order = d
            break

    if diff_order == 0:
        return original_series, 0
    else:
        return original_series.diff(diff_order).dropna(), diff_order

## FIT E PREDIZIONE

def auto_arima(series, max_p=2, max_q=2):
    """
    Auto ARIMA con selezione ordini candidati
    """
    try:
        if len(series) < 30:
            return (1, 1, 1)

        try:
            result = adfuller(series.dropna(), autolag='AIC', maxlag=5)
            is_stationary = result[1] < 0.05
            d = 0 if is_stationary else 1
        except:
            d = 1

        candidate_orders = [
            (1, d, 1),
            (0, d, 1),
            (1, d, 0),
            (2, d, 1),
            (1, d, 2),
            (0, d, 0),
        ]

        best_aic = np.inf
        best_order = (1, d, 1)

        for order in candidate_orders:
            try:
                model = ARIMA(series, order=order)
                fitted_model = model.fit(
                    method_kwargs={'warn_convergence': False},
                    low_memory=True
                )

                aic = fitted_model.aic
                if not np.isnan(aic) and aic < best_aic:
                    best_aic = aic
                    best_order = order

            except:
                continue

        return best_order

    except:
        return (1, 1, 1)

def fit_arima_predict(series, order=(1,1,1), steps=1):
    """
    ARIMA Fit e Predizione
    """
    try:
        if len(series) < 20:
            return series.iloc[-1] * 1.001, 0.5, 1000

        model = ARIMA(series, order=order)
        fitted_model = model.fit(
            method_kwargs={
                'warn_convergence': False,
                'maxiter': 10,
                'ftol': 1e-05
            },
            low_memory=True
        )

        forecast = fitted_model.forecast(steps=steps)
        predicted_price = forecast[0] if hasattr(forecast, '__len__') else forecast

        confidence = 0.6

        return predicted_price, confidence, fitted_model.aic

    except:
        try:
            drift = series.diff().tail(10).mean()
            predicted_price = series.iloc[-1] + drift
            return predicted_price, 0.4, 1000
        except:
            return series.iloc[-1], 0.3, 1000

def train_arima_predict_direction(window_data, auto_order=True):
    """
    Addestra ARIMA e predice la direzione del prossimo periodo
    """
    try:
        if len(window_data) < 50:
            print(f"Dati insufficienti ({len(window_data)} obs). Uso predizione di default.")
            return {
                'predicted_direction': 1,
                'expected_return': 0.001,
                'confidence': 0.5,
                'arima_order': (1,1,1),
                'aic': 1000
            }

        prices = window_data['Close']

        if auto_order:
            order = auto_arima(prices, max_p=2, max_q=2)
        else:
            order = (1, 1, 1)

        predicted_price, confidence, aic = fit_arima_predict(prices, order, steps=1)

        current_price = prices.iloc[-1]
        expected_return = (predicted_price - current_price) / current_price
        predicted_direction = 1 if expected_return > 0 else 0

        is_stationary, stationarity_p = check_stationarity(prices)

        print(f"ARIMA{order}, AIC: {aic:.2f}")
        print(f"Prezzo: {current_price:.2f} → {predicted_price:.2f}")
        print(f"Confidence: {confidence:.3f}, Stazionario: {is_stationary}")

        return {
            'predicted_direction': predicted_direction,
            'expected_return': expected_return,
            'confidence': confidence,
            'arima_order': order,
            'aic': aic,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'is_stationary': is_stationary,
            'stationarity_pvalue': stationarity_p
        }

    except Exception as e:
        print(f"Errore generale ARIMA: {e}")
        avg_return = window_data['Close'].pct_change().mean() if len(window_data) > 1 else 0.001
        return {
            'predicted_direction': 1 if avg_return > 0 else 0,
            'expected_return': avg_return,
            'confidence': 0.5,
            'arima_order': (1,1,1),
            'aic': 1000
        }



##      MAIN
def main():
    print("CONFIGURAZIONE WALK-FORWARD ARIMA ANALYSIS")
    print("="*60)

    data_SP500 = pd.read_csv('DATA/SP500Data.csv')
    data_SP500['Date'] = pd.to_datetime(data_SP500['Date'])
    data_SP500.set_index('Date', inplace=True)

    data_SP500 = data_SP500[data_SP500.index.dayofweek < 5]

    data_SP500 = data_SP500[data_SP500.index >= '2019-01-01']


    
    # infimo week per week
    #data_SP500_weekly = data_SP500.resample('W-FRI').last().dropna()

    window_size_years = 1.5 # anni
    window_size_days = int(window_size_years * 252) # per i trading days
    #window_size_weeks = int(window_size_years * 52) # per i trading weeks
    auto_order = True

    print(f"Dataset Totale: {len(data_SP500)} osservazioni dal {data_SP500.index[0].date()} al {data_SP500.index[-1].date()}")
    print(f"Window Size: {window_size_days} giorni (~{window_size_years} anni)")
    print(f"Predizioni Walk-Forward: {len(data_SP500) - window_size_days} osservazioni")
    print(f"ARIMA Mode: {'Auto-selection' if auto_order else 'Fixed (1,1,1)'}")
    print("="*60)

    walk_forward_results = walk_forward_arima_analysis(data_SP500, window_size_days, auto_order)

    calculate_metrics_and_visualize_walkforward(walk_forward_results)



## METRICHE E VISUALIZZAZIONE
def walk_forward_arima_analysis(data, window_size, auto_order=True):
    """
    Analisi walk-forward completa con ARIMA
    """
    mode_text = "AUTO-ARIMA" if auto_order else "ARIMA(1,1,1)"
    print(f"WALK-FORWARD {mode_text} ANALYSIS...")
    print(f"Periodo analisi: {data.index[window_size].date()} → {data.index[-1].date()}")
    print(f"Numero predizioni: {len(data) - window_size}")
    print("="*60)

    all_predictions = []

    for i in range(window_size, len(data)):
        window_data = data.iloc[i-window_size:i]
        current_date = data.index[i]

        print(f"\nPredizione #{i-window_size+1}/{len(data)-window_size}")
        print(f"Training Window: {window_data.index[0].strftime('%Y-%m-%d')} → {window_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Predicting: {current_date.strftime('%Y-%m-%d')}")

        prediction = train_arima_predict_direction(window_data, auto_order)

        actual_price = data.iloc[i]['Close']
        previous_price = data.iloc[i-1]['Close']
        actual_return = (actual_price - previous_price) / previous_price
        actual_direction = 1 if actual_return > 0 else 0

        result = {
            'date': current_date,
            'predicted_direction': prediction['predicted_direction'],
            'actual_direction': actual_direction,
            'predicted_return': prediction['expected_return'],
            'actual_return': actual_return,
            'confidence': prediction['confidence'],
            'arima_order': prediction['arima_order'],
            'aic': prediction['aic'],
            'predicted_price': prediction.get('predicted_price', previous_price),
            'actual_price': actual_price,
            'previous_price': previous_price,
            'is_stationary': prediction.get('is_stationary', False),
            'stationarity_pvalue': prediction.get('stationarity_pvalue', 1.0),
            'window_start': window_data.index[0],
            'window_end': window_data.index[-1]
        }

        all_predictions.append(result)

        print(f"ARIMA{prediction['arima_order']}, AIC: {prediction['aic']:.2f}")
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
            print(f"Accuracy ultime {len(recent_results)} predizioni: {recent_accuracy:.2f}%")
            print(f"AIC medio: {avg_aic:.2f}, Confidence media: {avg_confidence:.3f}")
            print(f"Velocità: {(time.time() - start_time)/(i-window_size+1):.3f}s per predizione")

        if 'start_time' not in locals():
            import time
            start_time = time.time()

    return all_predictions

def calculate_metrics_and_visualize_walkforward(all_predictions):
    """
    Calcola metriche e crea visualizzazioni
    """
    print("\n" + "="*60)
    print("RISULTATI FINALI ARIMA WALK-FORWARD DIRECTIONAL PREDICTION")
    print("="*60)

    if len(all_predictions) == 0:
        print("Nessuna predizione disponibile")
        return

    results_df = pd.DataFrame(all_predictions)

    total_predictions = len(results_df)
    correct_predictions = (results_df['predicted_direction'] == results_df['actual_direction']).sum()
    directional_accuracy = (correct_predictions / total_predictions) * 100

    print(f"\nPERFORMANCE COMPLESSIVA:")
    print(f"Periodo analizzato: {results_df['date'].min().date()} → {results_df['date'].max().date()}")
    print(f"Total predizioni: {total_predictions}")
    print(f"Predizioni corrette: {correct_predictions}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")

    up_predictions = results_df[results_df['predicted_direction'] == 1]
    down_predictions = results_df[results_df['predicted_direction'] == 0]

    up_correct = (up_predictions['predicted_direction'] == up_predictions['actual_direction']).sum()
    down_correct = (down_predictions['predicted_direction'] == down_predictions['actual_direction']).sum()

    up_accuracy = (up_correct / len(up_predictions) * 100) if len(up_predictions) > 0 else 0
    down_accuracy = (down_correct / len(down_predictions) * 100) if len(down_predictions) > 0 else 0

    print(f"\nBREAKDOWN PER DIREZIONE:")
    print(f"Predizioni UP: {len(up_predictions)} (Accuracy: {up_accuracy:.2f}%)")
    print(f"Predizioni DOWN: {len(down_predictions)} (Accuracy: {down_accuracy:.2f}%)")

    print(f"\nANALISI MODELLI ARIMA:")
    avg_confidence = results_df['confidence'].mean()
    avg_aic = results_df['aic'].mean()
    stationary_pct = (results_df['is_stationary']).mean() * 100

    order_counts = results_df['arima_order'].value_counts().head(5)

    print(f"Confidence media: {avg_confidence:.3f}")
    print(f"AIC medio: {avg_aic:.2f}")
    print(f"Serie stazionarie: {stationary_pct:.1f}%")
    print(f"Ordini ARIMA più usati:")
    for order, count in order_counts.items():
        print(f"  {order}: {count} volte ({count/total_predictions*100:.1f}%)")

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

    create_walkforward_visualizations(results_df)

    save_walkforward_results(results_df)

    if directional_accuracy > 52:
        print(f"\nRISULTATO: Modello ARIMA supera il caso casuale ({directional_accuracy:.2f}% > 50%)")
    elif directional_accuracy > 50:
        print(f"RISULTATO: Modello ARIMA leggermente migliore del caso casuale ({directional_accuracy:.2f}%)")
    else:
        print(f"RISULTATO: Modello ARIMA non supera il caso casuale ({directional_accuracy:.2f}% ≤ 50%)")

def create_walkforward_visualizations(results_df):
    """
    Crea visualizzazioni per i risultati walk-forward ARIMA
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0,0].plot(results_df['date'], results_df['cumulative_accuracy'], linewidth=2, color='blue', label='ARIMA Walk-Forward')
    axes[0,0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    axes[0,0].set_title('Directional Accuracy Cumulativa (ARIMA)')
    axes[0,0].set_ylabel('Accuracy (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)

    pred_counts = results_df['predicted_direction'].value_counts()
    axes[0,1].bar(['DOWN (0)', 'UP (1)'], [pred_counts.get(0, 0), pred_counts.get(1, 0)],
                  color=['red', 'green'], alpha=0.7)
    axes[0,1].set_title('Distribuzione Predizioni')
    axes[0,1].set_ylabel('Frequenza')
    for i, v in enumerate([pred_counts.get(0, 0), pred_counts.get(1, 0)]):
        axes[0,1].text(i, v + 10, str(v), ha='center', va='bottom')

    top_orders = results_df['arima_order'].value_counts().head(5)
    order_accuracies = []
    order_labels = []

    for order in top_orders.index:
        mask = results_df['arima_order'] == order
        if mask.sum() > 10:
            acc = (results_df[mask]['predicted_direction'] == results_df[mask]['actual_direction']).mean() * 100
            order_accuracies.append(acc)
            order_labels.append(f'ARIMA{order}')

    if order_accuracies:
        colors = ['purple', 'orange', 'green', 'brown', 'pink'][:len(order_accuracies)]
        bars = axes[0,2].bar(order_labels, order_accuracies, color=colors, alpha=0.7)
        axes[0,2].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        axes[0,2].set_title('Accuracy per Ordine ARIMA')
        axes[0,2].set_ylabel('Accuracy (%)')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars, order_accuracies):
            axes[0,2].text(bar.get_x() + bar.get_width()/2, acc + 1, f'{acc:.1f}%',
                          ha='center', va='bottom', fontsize=8)

    yearly_perf = results_df.groupby(results_df['date'].dt.year).apply(
        lambda x: (x['predicted_direction'] == x['actual_direction']).mean() * 100
    )
    axes[1,0].bar(yearly_perf.index.astype(str), yearly_perf.values, color='orange', alpha=0.7)
    axes[1,0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    axes[1,0].set_title('Performance Annuale')
    axes[1,0].set_ylabel('Accuracy (%)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)

    axes[1,1].scatter(results_df['predicted_return'], results_df['actual_return'], alpha=0.6, color='blue', s=20)
    min_val = min(results_df['predicted_return'].min(), results_df['actual_return'].min())
    max_val = max(results_df['predicted_return'].max(), results_df['actual_return'].max())
    axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
    axes[1,1].set_xlabel('Rendimento Predetto')
    axes[1,1].set_ylabel('Rendimento Reale')
    axes[1,1].set_title('Predetto vs Reale')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    window_size = min(50, len(results_df)//4)
    if len(results_df) > window_size:
        rolling_acc = results_df['predicted_direction'].rolling(window_size).apply(
            lambda x: (x == results_df['actual_direction'].iloc[x.index]).mean() * 100
        )
        axes[1,2].plot(results_df['date'], rolling_acc, linewidth=2, color='green')
        axes[1,2].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        axes[1,2].set_title(f'Rolling Accuracy ({window_size} predizioni)')
        axes[1,2].set_ylabel('Accuracy (%)')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('RESULTS/ARIMA_WalkForward_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualizzazione salvata in: RESULTS/ARIMA_WalkForward_Analysis.png")

def save_walkforward_results(results_df):
    """
    Salva i risultati walk-forward ARIMA in CSV
    """
    results_df.to_csv('RESULTS/ARIMA_WalkForward_Results.csv', index=False)
    print(f"Risultati ARIMA Walk-Forward salvati in: RESULTS/ARIMA_WalkForward_Results.csv")

if __name__ == "__main__":
    main()
    print("="*60)