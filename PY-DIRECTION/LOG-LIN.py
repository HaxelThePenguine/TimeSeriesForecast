import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="talk")


## FEATURE ENGINEERING E SELEZIONE

def create_features(data, lookback_days=20):
    """
    Crea features ingegnerizzate per la regressione
    """
    df = data.copy()

    # Calcola returns
    df['returns_1d'] = df['Close'].pct_change()
    df['returns_5d'] = df['Close'].pct_change(5)
    df['returns_10d'] = df['Close'].pct_change(10)
    df['returns_20d'] = df['Close'].pct_change(20)

    # Calcola medie mobili
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()

    # Calcola rapporto prezzo / media mobile
    df['price_to_sma5'] = df['Close'] / df['sma_5']
    df['price_to_sma10'] = df['Close'] / df['sma_10']
    df['price_to_sma20'] = df['Close'] / df['sma_20']
    df['sma5_to_sma20'] = df['sma_5'] / df['sma_20']

    # Calcola volatilità
    df['volatility_5d'] = df['returns_1d'].rolling(5).std()
    df['volatility_10d'] = df['returns_1d'].rolling(10).std()
    df['volatility_20d'] = df['returns_1d'].rolling(20).std()

    # Altre features 
    df['high_20d'] = df['High'].rolling(20).max()
    df['low_20d'] = df['Low'].rolling(20).min()
    df['price_position'] = (df['Close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'])

    if 'Volume' in df.columns:
        df['volume_sma_10'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
    else:
        df['volume_ratio'] = 1.0

    df['momentum_5d'] = df['Close'] / df['Close'].shift(5)
    df['momentum_10d'] = df['Close'] / df['Close'].shift(10)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs)) # Relative Strength Index pseudo magia non veramente utile ma interessante da vedere

    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns_1d'].shift(lag)
        df[f'volatility_lag_{lag}'] = df['volatility_5d'].shift(lag)

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df


## da miglioare con Lasso o simili non usato qua
def lasso_feature_selection(X, y):
    """
    Seleziona features usando LassoCV
    """
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X, y)
    selected_features = X.columns[(lasso.coef_ != 0)]
    return selected_features

def select_features(df):
    """
    Seleziona le features più importanti per il modello
    """
    feature_columns = [
        'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
        'price_to_sma5', 'price_to_sma10', 'price_to_sma20', 'sma5_to_sma20',
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'price_position', 'volume_ratio',
        'momentum_5d', 'momentum_10d', 'rsi',
        'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
        'volatility_lag_1', 'volatility_lag_2', 'volatility_lag_3', 'volatility_lag_5'
    ]
    # posso usare anche Lasso per selezionare le features da aggiustare
    # selection = lasso_feature_selection(df[feature_columns].dropna(), df['target'].dropna())

    available_features = [col for col in feature_columns if col in df.columns]

    return available_features



## main e predizione

def train_regression_predict_direction(window_data, use_logistic=True):
    """
    Addestra modello di regressione e predice la direzione del prossimo periodo
    """
    try:
        if len(window_data) < 50:
            return {
                'predicted_direction': 1,
                'expected_return': 0.001,
                'confidence': 0.5,
                'model_type': 'default'
            }

        df_features = create_features(window_data)
        feature_columns = select_features(df_features)

        df_clean = df_features[feature_columns + ['target']].dropna()

        if len(df_clean) < 30:
            recent_return = window_data['Close'].pct_change().tail(10).mean()
            return {
                'predicted_direction': 1 if recent_return > 0 else 0,
                'expected_return': recent_return,
                'confidence': 0.6,
                'model_type': 'trend'
            }

        X = df_clean[feature_columns].values
        y = df_clean['target'].values

        # Standardizza le features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Suddivide in train/validation
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]


        #scelta modello 
        if use_logistic:
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            #y_val_proba = model.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.array([0.5])

            X_last = X_scaled[-1].reshape(1, -1)
            predicted_direction = model.predict(X_last)[0]
            predicted_proba = model.predict_proba(X_last)[0]
            confidence = max(predicted_proba)

            model_type = 'logistic'

        else:
            model = LinearRegression()

            returns_target = df_clean['returns_1d'].shift(-1).dropna().values
            if len(returns_target) != len(X_scaled):
                returns_target = np.concatenate([returns_target, [returns_target[-1]]])

            returns_target = returns_target[:len(X_scaled)]

            model.fit(X_train, returns_target[:len(X_train)])

            X_last = X_scaled[-1].reshape(1, -1)
            predicted_return = model.predict(X_last)[0]
            predicted_direction = 1 if predicted_return > 0 else 0

            confidence = min(0.95, max(0.5, 0.5 + abs(predicted_return) * 10))

            model_type = 'linear'

        if len(y_val) > 0 and use_logistic:
            val_accuracy = accuracy_score(y_val, y_val_pred)
        else:
            val_accuracy = 0.5

        if use_logistic:
            expected_return = (confidence - 0.5) * 0.02 * (1 if predicted_direction == 1 else -1)
        else:
            expected_return = predicted_return

        return {
            'predicted_direction': int(predicted_direction),
            'expected_return': expected_return,
            'confidence': confidence,
            'model_type': model_type,
            'val_accuracy': val_accuracy,
            'n_features': len(feature_columns)
        }

    except Exception as e:
        avg_return = window_data['Close'].pct_change().mean() if len(window_data) > 1 else 0.001
        return {
            'predicted_direction': 1 if avg_return > 0 else 0,
            'expected_return': avg_return,
            'confidence': 0.5,
            'model_type': 'fallback'
        }

def main():
    print("CONFIGURAZIONE WALK-FORWARD REGRESSION ANALYSIS")
    print("="*60)

    data_SP500 = pd.read_csv('DATA/SP500Data.csv')
    data_SP500['Date'] = pd.to_datetime(data_SP500['Date'])
    data_SP500.set_index('Date', inplace=True)

    data_SP500 = data_SP500[data_SP500.index.dayofweek < 5]

    window_size_years = 2 ## circa 2 anni
    window_size_days = window_size_years * 252
    use_logistic = False

    model_name = "Logistic Regression" if use_logistic else "Linear Regression"

    print(f"Dataset Totale: {len(data_SP500)} osservazioni dal {data_SP500.index[0].date()} al {data_SP500.index[-1].date()}")
    print(f"Window Size: {window_size_days} giorni (~{window_size_years} anni)")
    print(f"Predizioni Walk-Forward: {len(data_SP500) - window_size_days} osservazioni")
    print(f"Model Type: {model_name}")
    print(f"Features: Returns, SMA ratios, Volatility, RSI, Momentum, Lags")
    print("="*60)
    # qui va day today walk forward
    walk_forward_results = walk_forward_regression_analysis(data_SP500, window_size_days, use_logistic)

    calculate_metrics_and_visualize_walkforward(walk_forward_results, model_name)

def walk_forward_regression_analysis(data, window_size, use_logistic=True):
    """
    Analisi walk-forward completa con Regression Models
    """
    model_name = "LOGISTIC" if use_logistic else "LINEAR"
    print(f"WALK-FORWARD {model_name} REGRESSION ANALYSIS...")
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

        prediction = train_regression_predict_direction(window_data, use_logistic)

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
            'model_type': prediction['model_type'],
            'val_accuracy': prediction.get('val_accuracy', 0.5),
            'n_features': prediction.get('n_features', 0),
            'actual_price': actual_price,
            'previous_price': previous_price,
            'window_start': window_data.index[0],
            'window_end': window_data.index[-1]
        }

        all_predictions.append(result)

        print(f"Model: {prediction['model_type'].upper()}, Confidence: {prediction['confidence']:.3f}")
        print(f"Rendimento atteso: {prediction['expected_return']:.4f}")
        print(f"Direzione predetta: {'UP' if prediction['predicted_direction'] == 1 else 'DOWN'}")
        print(f"Direzione reale: {'UP' if actual_direction == 1 else 'DOWN'}")
        print(f"Corretto: {'SI' if prediction['predicted_direction'] == actual_direction else 'NO'}")

        if (i - window_size + 1) % 50 == 0:
            recent_results = all_predictions[-50:]
            recent_accuracy = np.mean([r['predicted_direction'] == r['actual_direction'] for r in recent_results]) * 100
            avg_confidence = np.mean([r['confidence'] for r in recent_results])
            print(f"Accuracy ultime 50 predizioni: {recent_accuracy:.2f}%")
            print(f"Confidence media: {avg_confidence:.3f}")

    return all_predictions


## METRICHE E VISUALIZZAZIONI WALK-FORWARD

def calculate_metrics_and_visualize_walkforward(all_predictions, model_name):
    """
    Calcola metriche e crea visualizzazioni per i risultati walk-forward Regression
    """
    print("\n" + "="*60)
    print(f"RISULTATI FINALI {model_name.upper()} WALK-FORWARD DIRECTIONAL PREDICTION")
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

    print(f"\nANALISI MODELLO {model_name.upper()}:")
    avg_confidence = results_df['confidence'].mean()
    avg_val_accuracy = results_df['val_accuracy'].mean()
    avg_features = results_df['n_features'].mean()

    model_counts = results_df['model_type'].value_counts()

    print(f"Confidence media: {avg_confidence:.3f}")
    print(f"Val Accuracy media: {avg_val_accuracy:.3f}")
    print(f"Features medie: {avg_features:.1f}")
    print(f"Tipi di modello usati:")
    for model_type, count in model_counts.items():
        print(f"  {model_type}: {count} predizioni ({count/total_predictions*100:.1f}%)")

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
        print(f"{year}: {acc:.2f}% ({year_count} predizioni)")

    #create_walkforward_visualizations(results_df, model_name)

    #save_walkforward_results(results_df, model_name)

    if directional_accuracy > 52:
        print(f"\nRISULTATO: Modello {model_name} supera il caso casuale ({directional_accuracy:.2f}% > 50%)")
    elif directional_accuracy > 50:
        print(f"RISULTATO: Modello {model_name} leggermente migliore del caso casuale ({directional_accuracy:.2f}%)")
    else:
        print(f"RISULTATO: Modello {model_name} non supera il caso casuale ({directional_accuracy:.2f}% ≤ 50%)")

    # no visualizzazioni e salvataggi per ora
if __name__ == "__main__":
    main()
    