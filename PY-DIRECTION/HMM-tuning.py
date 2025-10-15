import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Funzione di test per visualizzare gli stati nascosti
def scatter_with_states(dates, returns, states, n_states):
    plt.figure(figsize=(10, 6))
    for state in range(n_states):
        state_data = returns[states == state]
        plt.scatter(dates[states == state], state_data, label=f'Stato {state}')
    plt.title('Stati Nascosti HMM')
    plt.xlabel('Data')
    plt.ylabel('Log Returns')
    plt.legend()
    plt.show()

data_SP500 = pd.read_csv('DATA/SP500Data.csv')
data_SP500['Date'] = pd.to_datetime(data_SP500['Date'])
data_SP500.set_index('Date', inplace=True)

data_SP500_weekly = data_SP500.resample('W-FRI').last()

data_SP500_weeklyReturns = np.log(data_SP500_weekly['Close'] / data_SP500_weekly['Close'].shift(1).dropna()).to_frame(name = 'LogReturns').dropna()

def train_hmm_predict_direction(window_data, n_states=2):
    """
    Addestra HMM e predice la direzione del prossimo periodo
    """
    try:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(window_data[['LogReturns']])

        model = hmm.GMMHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=0,n_mix=3)
        model.fit(window_data[['LogReturns']])

        if hasattr(model, 'startprob_') and np.any(np.isnan(model.startprob_)):
            print("Modello HMM non valido (startprob_ NaN), ritorno predizione casuale")
            return {
                'predicted_direction': 1 if np.random.random() > 0.5 else 0,
                'expected_return': 0.0,
                'most_likely_state': 0,
                'state_probability': 0.5,
                'state_returns': {0: 0.0, 1: 0.0},
                'hidden_states': np.zeros(len(window_data))
            }

        posteriors = model.predict_proba(window_data[['LogReturns']])

        pi_next = posteriors[-1] @ model.transmat_

        hidden_states = model.predict(train_data)

        window_data_copy = window_data.copy()
        window_data_copy['HiddenState'] = hidden_states

        state_returns = window_data_copy.groupby('HiddenState')['LogReturns'].mean()

        most_likely_next_state = np.argmax(pi_next)
        next_state_prob = pi_next
        print(f"Most likely state: {most_likely_next_state}")
        print(f"Available states: {state_returns.index.tolist()}")

        if most_likely_next_state not in state_returns.index:
            print(f"Stato {most_likely_next_state} non trovato! Uso stato {state_returns.index[0]}")
            most_likely_next_state = state_returns.index[0]

        expected_return = state_returns[most_likely_next_state]

        predicted_direction = 1 if expected_return > 0 else 0

        return {
            'predicted_direction': predicted_direction,
            'expected_return': expected_return,
            'most_likely_state': most_likely_next_state,
            'state_probability': next_state_prob[most_likely_next_state],
            'state_returns': state_returns.to_dict(),
            'hidden_states': hidden_states
        }

    except Exception as e:
        print(f"Errore nell'addestramento HMM: {e}")
        return {
            'predicted_direction': 1 if np.random.random() > 0.5 else 0,
            'expected_return': 0.0,
            'most_likely_state': 0,
            'state_probability': 0.5,
            'state_returns': {0: 0.0, 1: 0.0},
            'hidden_states': np.zeros(len(window_data))
        }

n_states = 2 ## numero di stati HMM
window_size = 104 ## circa 2 anni

print("CONFIGURAZIONE WALK-FORWARD HMM ANALYSIS")
print(f"Dataset Totale: {len(data_SP500_weeklyReturns)} osservazioni dal {data_SP500_weeklyReturns.index[0].date()} al {data_SP500_weeklyReturns.index[-1].date()}")
print(f"Window Size: {window_size} settimane (~2 anni)")
print(f"Predizioni Walk-Forward: {len(data_SP500_weeklyReturns) - window_size} osservazioni")
print(f"Stati HMM: {n_states}")
print("="*60)

def walk_forward_hmm_analysis(data, window_size, n_states):
    """
    Analisi walk-forward completa su tutto il dataset
    """
    print("WALK-FORWARD HMM ANALYSIS...")
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

        prediction = train_hmm_predict_direction(window_data, n_states)

        actual_return = data.iloc[i]['LogReturns']
        actual_direction = 1 if actual_return > 0 else 0

        result = {
            'date': current_date,
            'predicted_direction': prediction['predicted_direction'],
            'actual_direction': actual_direction,
            'predicted_return': prediction['expected_return'],
            'actual_return': actual_return,
            'most_likely_state': prediction['most_likely_state'],
            'state_probability': prediction['state_probability'],
            'state_returns': prediction['state_returns'],
            'hidden_states_count': len(np.unique(prediction['hidden_states'])),
            'window_start': window_data.index[0],
            'window_end': window_data.index[-1]
        }

        all_predictions.append(result)

        print(f"Stato predetto: {prediction['most_likely_state']} (prob: {prediction['state_probability']:.3f})")
        print(f"Rendimento atteso: {prediction['expected_return']:.4f}")
        print(f"Direzione predetta: {'UP' if prediction['predicted_direction'] == 1 else 'DOWN'}")
        print(f"Direzione reale: {'UP' if actual_direction == 1 else 'DOWN'}")
        print(f"Corretto: {'SI' if prediction['predicted_direction'] == actual_direction else 'NO'}")

        if (i - window_size + 1) % 20 == 0:
            print(f"Rendimenti per stato: {prediction['state_returns']}")

    return all_predictions

walk_forward_results = walk_forward_hmm_analysis(data_SP500_weeklyReturns, window_size, n_states)

## METRICHE E VISUALIZZAZIONI WALK-FORWARD
def calculate_metrics_and_visualize_walkforward(walk_forward_results, n_states):
    """
    Calcola metriche e crea visualizzazioni per i risultati walk-forward
    """
    print("\n" + "="*60)
    print("RISULTATI FINALI HMM WALK-FORWARD DIRECTIONAL PREDICTION")
    print("="*60)

    if len(walk_forward_results) == 0:
        print("Nessun risultato disponibile!")
        return

    results_df = pd.DataFrame(walk_forward_results)

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

    print(f"\nACCURACY PER STATO HMM:")
    for state in range(n_states):
        state_data = results_df[results_df['most_likely_state'] == state]
        if len(state_data) > 0:
            state_correct = (state_data['predicted_direction'] == state_data['actual_direction']).sum()
            state_accuracy = (state_correct / len(state_data)) * 100
            avg_return = state_data['predicted_return'].mean()
            print(f"Stato {state}: {state_accuracy:.2f}% (Freq: {len(state_data)}, Ret medio: {avg_return:.4f})")

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
        print(f"{year}: {acc:.2f}% ({year_count} predizioni)")

    create_walkforward_visualizations(results_df)

    save_walkforward_results(results_df)

    if directional_accuracy > 52:
        print(f"\nRISULTATO: Modello supera il caso casuale ({directional_accuracy:.2f}% > 50%)")
    elif directional_accuracy > 50:
        print(f"RISULTATO: Modello leggermente migliore del caso casuale ({directional_accuracy:.2f}%)")
    else:
        print(f"RISULTATO: Modello non supera il caso casuale ({directional_accuracy:.2f}% ≤ 50%)")

def create_walkforward_visualizations(results_df):
    """
    Crea visualizzazioni specifiche per i risultati walk-forward
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0,0].plot(results_df['date'], results_df['cumulative_accuracy'], linewidth=2, color='blue')
    axes[0,0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    axes[0,0].set_title('Directional Accuracy Cumulativa')
    axes[0,0].set_ylabel('Accuracy (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)

    pred_counts = results_df['predicted_direction'].value_counts()
    axes[0,1].bar(['DOWN (0)', 'UP (1)'], [pred_counts.get(0, 0), pred_counts.get(1, 0)],
                  color=['red', 'green'], alpha=0.7)
    axes[0,1].set_title('Distribuzione Predizioni')
    axes[0,1].set_ylabel('Frequenza')

    states = results_df['most_likely_state'].unique()
    state_accs = []
    state_labels = []
    for state in sorted(states):
        state_data = results_df[results_df['most_likely_state'] == state]
        if len(state_data) > 0:
            acc = (state_data['predicted_direction'] == state_data['actual_direction']).mean() * 100
            state_accs.append(acc)
            state_labels.append(f'Stato {state}')

    if state_accs:
        axes[0,2].bar(state_labels, state_accs, color='purple', alpha=0.7)
        axes[0,2].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        axes[0,2].set_title('Accuracy per Stato HMM')
        axes[0,2].set_ylabel('Accuracy (%)')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)

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

    axes[1,1].scatter(results_df['predicted_return'], results_df['actual_return'], alpha=0.6, color='orange')
    axes[1,1].plot([-0.1, 0.1], [-0.1, 0.1], 'r--', alpha=0.7, label='Perfect Prediction')
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
    plt.savefig('RESULTS/HMM_WalkForward_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualizzazione salvata in: RESULTS/HMM_WalkForward_Analysis.png")

def save_walkforward_results(all_predictions):
    """
    Salva i risultati walk-forward in CSV
    """
    if len(all_predictions) > 0:
        results_df = pd.DataFrame(all_predictions)
        results_df.to_csv('RESULTS/HMM_WalkForward_Results.csv', index=False)
        print(f"Risultati Walk-Forward salvati in: RESULTS/HMM_WalkForward_Results.csv")

calculate_metrics_and_visualize_walkforward(walk_forward_results, n_states)

