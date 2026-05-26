import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def prepare_training_data(df: pd.DataFrame, early_turns: int = 7) -> pd.DataFrame:
    """
    Transforms the turn-by-turn telemetry dataset into a session-level dataset.
    Extracts features from only the first N turns to predict the total session cost.
    """
    print(f"Aggregating dataset to session level using first {early_turns} turns...")
    
    # 1. Calculate Target Variable (Total Cost per Session)
    target_df = df.groupby('session_id')['turn_cost'].sum().reset_index()
    target_df.rename(columns={'turn_cost': 'total_session_cost'}, inplace=True)
    
    # 2. Extract Early Features (Only data from turn <= early_turns)
    early_df = df[df['turn_number'] <= early_turns].copy()
    
    # Calculate initial token load (sum of tokens in early turns)
    feature_df = early_df.groupby('session_id')['input_tokens'].sum().reset_index()
    feature_df.rename(columns={'input_tokens': 'early_input_tokens'}, inplace=True)
    
    # Count any errors encountered in the early turns
    errors_df = early_df.groupby('session_id')['has_error'].sum().reset_index()
    errors_df.rename(columns={'has_error': 'early_error_count'}, inplace=True)
    
    # Extract the model used (assumes 1 model per session)
    model_df = early_df.groupby('session_id')['model'].first().reset_index()
    
    # 3. Merge everything together
    ml_df = target_df.merge(feature_df, on='session_id')
    ml_df = ml_df.merge(errors_df, on='session_id')
    ml_df = ml_df.merge(model_df, on='session_id')
    
    # 4. One-Hot Encode the Categorical 'model' column
    ml_df = pd.get_dummies(ml_df, columns=['model'], drop_first=False)
    
    return ml_df

def train_xgboost_model(ml_df: pd.DataFrame) -> xgb.XGBRegressor:
    """
    Trains an XGBoost Regressor to predict the total session cost.
    """
    # Define Features (X) and Target (y)
    # Drop session_id as it's not a predictive feature
    X = ml_df.drop(columns=['session_id', 'total_session_cost'])
    y = ml_df['total_session_cost']
    
    # Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining XGBoost Regressor on {len(X_train)} sessions...")
    
    # Initialize and Train Model
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=4, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Mean Absolute Error (MAE): ${mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
    print(f"R-Squared (R2 Score): {r2:.4f}")
    
    if r2 > 0.7:
        print("Conclusion: Model has strong predictive power!")
    elif r2 > 0.5:
        print("Conclusion: Model is decent, but could use more features (e.g., latency).")
    else:
        print("Conclusion: Model is struggling. Context growth may be too unpredictable.")

    return model

if __name__ == "__main__":
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_CSV_PATH = os.path.abspath(os.path.join(script_dir, "../../data/processed/processed_agentic_traces.csv"))
    
    # 2. Load the processed data
    print("Loading processed telemetry data...")
    df = pd.read_csv(PROCESSED_CSV_PATH)
    
    # 3. Prepare data for ML (Using first 3 turns)
    ml_dataframe = prepare_training_data(df, early_turns=3)
    
    # 4. Train and Evaluate
    trained_model = train_xgboost_model(ml_dataframe)