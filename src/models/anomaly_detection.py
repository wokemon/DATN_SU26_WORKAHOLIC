import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def prepare_sequential_data(df: pd.DataFrame, sequence_length: int = 3):
    """
    Transforms flat tabular data into 3D sequential data (Samples, Time Steps, Features)
    required for LSTM neural networks. Groups strictly by session_id to prevent data leakage.
    """
    print(f"Structuring data into sliding windows of {sequence_length} turns...")
    
    # Sort chronologically to ensure time-series integrity
    df = df.sort_values(by=['session_id', 'turn_number'])
    
    # Select our time-series features
    # 'pre_gap' is our latency metric (and our target)
    features = ['input_tokens', 'output_length', 'turn_cost', 'pre_gap']
    
    # Scale the features (LSTMs are highly sensitive to unscaled data)
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    X, y = [], []
    
    # We must process sequence sliding windows per session so we don't 
    # accidentally use Turn 50 of Session A to predict Turn 1 of Session B
    grouped = df.groupby('session_id')
    
    for session_id, group in grouped:
        group_data = group[features].values
        
        # If a session is shorter than our sequence length, skip it
        if len(group_data) <= sequence_length:
            continue
            
        # Create sliding windows
        for i in range(len(group_data) - sequence_length):
            # The historical window (e.g., Turns 1, 2, 3)
            window = group_data[i : i + sequence_length]
            # The target to predict (e.g., pre_gap of Turn 4)
            # Index 3 corresponds to 'pre_gap' in our features list
            target_latency = group_data[i + sequence_length, 3] 
            
            X.append(window)
            y.append(target_latency)
            
    return np.array(X), np.array(y), scaler

def build_and_train_lstm(X_train, y_train, X_test, y_test, sequence_length: int, num_features: int):
    """
    Constructs and trains the Long Short-Term Memory (LSTM) Neural Network.
    """
    print(f"\nInitializing LSTM Architecture (Input Shape: {sequence_length} timesteps, {num_features} features)...")
    
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2), # Prevents the neural network from memorizing the training data
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1) # Output layer: A single continuous value (Predicted pre_gap latency)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("Training LSTM (Epochs: 20)...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print("\nEvaluating Model on Test Data...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Set Mean Absolute Error (Scaled): {mae:.4f}")
    
    return model

def detect_anomalies(model, X, y_actual, scaler):
    """
    Uses the trained LSTM to forecast latency. If the actual latency is wildly higher 
    than the forecast, it flags the step as an 'Anomaly' (Performance Bottleneck).
    """
    print("\nRunning Anomaly Detection...")
    
    # Predict the expected latency for all sequences
    y_pred = model.predict(X).flatten()
    
    # Calculate the residual error (Difference between actual and predicted)
    errors = np.abs(y_actual - y_pred)
    
    # Dynamic Anomaly Threshold: Mean error + 3 Standard Deviations
    # Mathematically isolates the extreme outliers (bottlenecks)
    threshold = np.mean(errors) + (3 * np.std(errors))
    print(f"Dynamic Anomaly Threshold calculated at: {threshold:.4f} (Scaled Error)")
    
    # Flag anomalies
    anomalies = errors > threshold
    anomaly_count = np.sum(anomalies)
    anomaly_percentage = (anomaly_count / len(y_actual)) * 100
    
    print("\n--- Latency Bottleneck Report ---")
    print(f"Total Turn Sequences Analyzed: {len(y_actual)}")
    print(f"Severe Latency Spikes Detected: {anomaly_count} ({anomaly_percentage:.2f}% of all turns)")
    print("Conclusion: These flagged turns represent critical points where the AI Agent's tool execution stalled.")

if __name__ == "__main__":
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_CSV_PATH = os.path.abspath(os.path.join(script_dir, "../../data/processed/processed_agentic_traces.csv"))
    
    # 2. Load the processed data
    print("Loading telemetry data for sequence modeling...")
    df = pd.read_csv(PROCESSED_CSV_PATH)
    
    # 3. Create 3D Sequences (Look back 3 turns to predict the 4th)
    SEQ_LENGTH = 3
    X, y, scaler = prepare_sequential_data(df, sequence_length=SEQ_LENGTH)
    
    # 4. Train/Test Split (80/20) - Keep sequential order roughly intact by splitting arrays directly
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 5. Train Model
    trained_lstm = build_and_train_lstm(X_train, y_train, X_test, y_test, SEQ_LENGTH, X.shape[2])
    
    # 6. Detect Performance Bottlenecks
    detect_anomalies(trained_lstm, X_test, y_test, scaler)