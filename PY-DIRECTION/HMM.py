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

data_SP500_weeklyReturns.dropna(inplace=True)

def train_hmm_predict_direction(window_data, n_states=2):
    """
    Addestra HMM e predice la direzione del prossimo periodo
    """
    try:
        returns_data = window_data[['LogReturns']]

        if len(returns_data) < 10:
            print(f"Dati insufficienti ({len(returns_data)} obs). Uso predizione di default.")
            return {
                'predicted_direction': 1,
                'expected_return': 0.001,
                'most_likely_state': 0,
                'state_probability': 0.5,
                'state_returns': {0: 0.001, 1: 0.001},
                'hidden_states': np.zeros(len(window_data))
            }

        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_data) # normalizzazione

        q1, q3 = np.percentile(scaled_returns, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        scaled_returns = np.clip(scaled_returns, lower_bound, upper_bound)

        # specifiche utilizzate, le possiamo modificare se necessario
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            tol=1e-6,
            random_state=42,
            verbose=False
        )

        try:
            model.fit(scaled_returns)
        except Exception as e:
            print(f"Errore training HMM: {e}. Uso modello semplificato.")
            avg_return = window_data['LogReturns'].mean()
            return {
                'predicted_direction': 1 if avg_return > 0 else 0,
                'expected_return': avg_return,
                'most_likely_state': 0,
                'state_probability': 0.5,
                'state_returns': {i: avg_return for i in range(n_states)},
                'hidden_states': np.zeros(len(window_data))
            }

        try:
            hidden_states = model.predict(scaled_returns)
        except Exception as e:
            print(f"Errore predizione stati: {e}")
            hidden_states = np.zeros(len(window_data))

        window_data_copy = window_data.copy()
        window_data_copy['HiddenState'] = hidden_states

        state_returns = window_data_copy.groupby('HiddenState')['LogReturns'].mean()

        state_returns_dict = {}
        for state in range(n_states):
            if state in state_returns.index:
                state_returns_dict[state] = state_returns[state]
            else:
                state_returns_dict[state] = window_data['LogReturns'].mean()
                print(f"Stato {state} non trovato! Uso media: {state_returns_dict[state]:.4f}")

        min_obs_per_state = 5
        valid_states = {}
        for state in range(n_states):
            state_count = np.sum(hidden_states == state)
            if state_count >= min_obs_per_state:
                valid_states[state] = state_returns_dict[state]
            else:
                print(f"Stato {state} ha solo {state_count} osservazioni, rimosso")

        if len(valid_states) < 2:
            print("Troppi pochi stati validi, uso predizione basata su trend")
            recent_returns = window_data['LogReturns'].tail(20).mean() # semplice media ultimi 20, tipo fall back
            return {
                'predicted_direction': 1 if recent_returns > 0 else 0,
                'expected_return': recent_returns,
                'most_likely_state': 0,
                'state_probability': 0.6,
                'state_returns': {0: recent_returns},
                'hidden_states': hidden_states
            }

        try:
            posteriors = model.predict_proba(scaled_returns)
            pi_next = posteriors[-1] @ model.transmat_
            most_likely_next_state = np.argmax(pi_next)
            state_probability = pi_next[most_likely_next_state]

            if most_likely_next_state not in valid_states:
                recent_trend = window_data['LogReturns'].tail(10).mean()
                best_state = min(valid_states.keys(),
                               key=lambda s: abs(valid_states[s] - recent_trend))
                most_likely_next_state = best_state
                state_probability = 0.6
                print(f"Stato predetto non valido, uso stato {best_state}")

        except Exception as e:
            print(f"Errore calcolo probabilità: {e}")
            most_likely_next_state = list(valid_states.keys())[0]
            state_probability = 0.5

        expected_return = valid_states.get(most_likely_next_state, window_data['LogReturns'].mean())

        predicted_direction = 1 if expected_return > 0 else 0

        print(f"Stati validi: {list(valid_states.keys())}")
        print(f"Stato predetto: {most_likely_next_state}")

        return {
            'predicted_direction': predicted_direction,
            'expected_return': expected_return,
            'most_likely_state': most_likely_next_state,
            'state_probability': state_probability,
            'state_returns': valid_states,
            'hidden_states': hidden_states
        }

    except Exception as e:
        print(f"Errore generale HMM: {e}")
        avg_return = window_data['LogReturns'].mean() if len(window_data) > 0 else 0.001
        return {
            'predicted_direction': 1 if avg_return > 0 else 0,
            'expected_return': avg_return,
            'most_likely_state': 0,
            'state_probability': 0.5,
            'state_returns': {0: avg_return, 1: avg_return},
            'hidden_states': np.zeros(len(window_data))
        }
# configurazione walk-forward
n_states = 3 ## numero di stati HMM
window_size = 156 ## circa 3 anni

print("CONFIGURAZIONE WALK-FORWARD HMM ANALYSIS")
print(f"Dataset Totale: {len(data_SP500_weeklyReturns)} osservazioni dal {data_SP500_weeklyReturns.index[0].date()} al {data_SP500_weeklyReturns.index[-1].date()}")
print(f"Window Size: {window_size} settimane (~3 anni)")
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

    #create_walkforward_visualizations(results_df)

    #save_walkforward_results(results_df)

    if directional_accuracy > 52:
        print(f"\nRISULTATO: Modello supera il caso casuale ({directional_accuracy:.2f}% > 50%)")
    elif directional_accuracy > 50:
        print(f"RISULTATO: Modello leggermente migliore del caso casuale ({directional_accuracy:.2f}%)")
    else:
        print(f"RISULTATO: Modello non supera il caso casuale ({directional_accuracy:.2f}% ≤ 50%)")

    # no visualizzazioni e salvataggi per ora
    
calculate_metrics_and_visualize_walkforward(walk_forward_results, n_states)

