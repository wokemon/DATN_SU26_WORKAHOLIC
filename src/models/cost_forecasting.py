import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --- 1. DEFINE API PRICING ---
# Price per 1 input token (e.g., $3.00 per 1M tokens = 0.000003)
# Update these to match the exact rates used in your dataset!
MODEL_PRICING = {
    'claude-opus-4-6': 15.0 / 1_000_000,
    'claude-sonnet-4-6': 3.0 / 1_000_000,
    'deepseek-v3.1': 0.14 / 1_000_000, 
    'minimax-m2.5': 0.10 / 1_000_000   
}

def get_token_price(model_name: str) -> float:
    """Helper to safely fetch the token price, defaulting to Sonnet if unknown."""
    return MODEL_PRICING.get(model_name, 3.0 / 1_000_000)

def prepare_training_data(df: pd.DataFrame, early_turns: int = 7) -> pd.DataFrame:
    """
    Transforms the turn-by-turn telemetry dataset into a session-level dataset.
    Extracts token trajectory features to predict total session TOKENS, not cost.
    """
    print(f"Aggregating dataset to session level using first {early_turns} turns...")
    
    # 1. Target Variable: Total Input Tokens (Decoupled from pricing!)
    target_df = df.groupby('session_id')['input_tokens'].sum().reset_index()
    target_df.rename(columns={'input_tokens': 'total_session_tokens'}, inplace=True)
    
    # 2. Extract Early Features
    early_df = df[df['turn_number'] <= early_turns].copy()
    
    # A. Task Source Extraction
    task_df = early_df.groupby('session_id')['session_id'].first().reset_index(name='task_source')
    task_df['task_source'] = task_df['session_id'].apply(lambda x: x.split('__')[0] if '__' in x else x.split('_')[0])
    
    # B. Cumulative Early Tokens & Errors
    feature_df = early_df.groupby('session_id').agg(
        early_input_tokens=('input_tokens', 'sum'),
        early_error_count=('has_error', 'sum'),
        early_avg_pre_gap=('pre_gap', 'mean')
    ).reset_index()
    
    # C. Token Growth Rate (The Mathematical Trajectory)
    first_turn = early_df[early_df['turn_number'] == 1][['session_id', 'input_tokens']].rename(columns={'input_tokens': 'tokens_turn_1'})
    last_turn = early_df.sort_values('turn_number').groupby('session_id').tail(1)[['session_id', 'turn_number', 'input_tokens']]
    last_turn.rename(columns={'input_tokens': 'tokens_last_early', 'turn_number': 'last_turn_num'}, inplace=True)
    
    growth_df = first_turn.merge(last_turn, on='session_id')
    growth_df['token_growth_rate'] = (growth_df['tokens_last_early'] - growth_df['tokens_turn_1']) / np.maximum(1, (growth_df['last_turn_num'] - 1))
    growth_df = growth_df[['session_id', 'token_growth_rate']]
    
    # D. Model Type (Keep original for price mapping, duplicate for one-hot encoding)
    model_df = early_df.groupby('session_id')['model'].first().reset_index()
    model_df['original_model'] = model_df['model'] # Save for later cost calculation
    
    # 3. Merge everything together
    ml_df = target_df.merge(feature_df, on='session_id')
    ml_df = ml_df.merge(growth_df, on='session_id')
    ml_df = ml_df.merge(task_df[['session_id', 'task_source']], on='session_id')
    ml_df = ml_df.merge(model_df, on='session_id')
    
    # 4. One-Hot Encode (Keep 'original_model' safe)
    ml_df = pd.get_dummies(ml_df, columns=['model', 'task_source'], drop_first=False)
    
    return ml_df

def train_xgboost_model(ml_df: pd.DataFrame) -> xgb.XGBRegressor:
    """
    Trains an XGBoost Regressor to predict total tokens, then calculates final cost metrics.
    """
    # Isolate the original model names for cost conversion later, then drop non-features
    original_models = ml_df['original_model']
    X = ml_df.drop(columns=['session_id', 'total_session_tokens', 'original_model'])
    
    # APPLY LOG TRANSFORM TO TOKENS
    y_log = np.log1p(ml_df['total_session_tokens'])
    
    # Train/Test Split (Include original_models in the split to map costs later)
    X_train, X_test, y_train_log, y_test_log, _, models_test = train_test_split(
        X, y_log, original_models, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining XGBoost Regressor on {len(X_train)} sessions...")
    print(f"Target: Total Session Tokens (Decoupled from pricing)")
    
    # Initialize and Train Model
    model = xgb.XGBRegressor(
        n_estimators=200,    
        learning_rate=0.03,  
        max_depth=5, 
        subsample=0.8,
        colsample_bytree=0.8, 
        random_state=42
    )
    
    # Train on logarithmic tokens
    model.fit(X_train, y_train_log)
    
    # PREDICT AND REVERSE TRANSFORM (Back to raw token counts)
    y_pred_log_tokens = model.predict(X_test)
    y_pred_tokens = np.expm1(y_pred_log_tokens)
    y_test_tokens_real = np.expm1(y_test_log)
    
    # --- RE-COUPLE WITH PRICING FOR FINAL METRICS ---
    # Convert real and predicted tokens back to dollars using our dictionary
    price_multipliers = models_test.apply(get_token_price).values
    
    y_test_cost = y_test_tokens_real * price_multipliers
    y_pred_cost = y_pred_tokens * price_multipliers
    
    # Evaluation Metrics (Dollars)
    mae_cost = mean_absolute_error(y_test_cost, y_pred_cost)
    rmse_cost = np.sqrt(mean_squared_error(y_test_cost, y_pred_cost))
    r2_cost = r2_score(y_test_cost, y_pred_cost)
    
    # Evaluation Metrics (Tokens - Pure Agent Behavior)
    r2_tokens = r2_score(y_test_tokens_real, y_pred_tokens)
    
    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Token Prediction R-Squared: {r2_tokens:.4f} (How well it learned agent behavior)")
    print(f"Cost Prediction R-Squared:  {r2_cost:.4f} (Final business metric)")
    print(f"Mean Absolute Error (MAE):  ${mae_cost:.4f}")
    print(f"Root Mean Squared Error:    ${rmse_cost:.4f}")
    
    # Print Feature Importances
    importances = model.feature_importances_
    feature_names = X.columns
    print("\n--- Top 3 Feature Importances ---")
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(3)
    for index, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    return model

if __name__ == "__main__":
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_CSV_PATH = os.path.abspath(os.path.join(script_dir, "../../data/processed/processed_agentic_traces.csv"))
    
    # 2. Load the processed data
    print("Loading processed telemetry data...")
    df = pd.read_csv(PROCESSED_CSV_PATH)
    
    # 3. Prepare data for ML (Using first 7 turns)
    ml_dataframe = prepare_training_data(df, early_turns=7)
    
    # 4. Train and Evaluate
    trained_model = train_xgboost_model(ml_dataframe)