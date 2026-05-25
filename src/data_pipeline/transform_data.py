import os
import pandas as pd

# Define approximate pricing per 1 million tokens (Input, Output) in USD
# Why: Hardcoding a pricing dictionary avoids external API calls during the transformation 
# phase and allows for fast vectorized cost calculations.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "minimax-m2.5": (0.15, 1.15),
    "deepseek-v3.1": (0.21, 0.28),
    "default": (1.00, 2.00) # Fallback pricing
}

def calculate_turn_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculates the cost of a single turn based on model pricing.
    """
    prices = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    input_price_per_token: float = prices[0] / 1_000_000
    output_price_per_token: float = prices[1] / 1_000_000
    
    return (input_tokens * input_price_per_token) + (output_tokens * output_price_per_token)

def process_agentic_traces(input_filepath: str, output_filepath: str) -> None:
    """
    Loads raw telemetry data, extracts performance/cost features, and exports a lightweight CSV.
    """
    print(f"Loading heavy dataset from {input_filepath}...")
    print("Grab a coffee, this might take a minute or two to load into RAM...")
    
    # Load the data
    df: pd.DataFrame = pd.read_csv(input_filepath)
    
    print("Data loaded! Starting transformations...")
    
    # 1. Sequence Mapping: Calculate the turn_number
    # Why: We need sequential order per session to forecast latency spikes with LSTM later.
    df['turn_number'] = df.groupby('session_id').cumcount() + 1
    
    # 2. Token Counting: Estimate input tokens
    # Why: A strict 1:4 character-to-token ratio heuristic avoids the massive memory overhead 
    # of loading a Hugging Face tokenizer for 2.5GB of text.
    df['input_tokens'] = (df['input'].astype(str).str.len() // 4).astype(int)
    
    # 3. Cost Calculation
    # Why: Applying a lambda function row-by-row maps the specific model pricing to the token counts.
    df['turn_cost'] = df.apply(
        lambda row: calculate_turn_cost(
            model=row['model'], 
            input_tokens=row['input_tokens'], 
            output_tokens=row['output_length']
        ), 
        axis=1
    )
    
    # 4. Drop the massive 'input' column
    # Why: We have extracted the metadata we need. Carrying the raw text arrays forward 
    # will crash Power BI and slow down Scikit-Learn/XGBoost models.
    df_clean: pd.DataFrame = df.drop(columns=['input'])
    
    # Ensure output directory exists
    output_dir: str = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Export the processed data
    print(f"Saving processed data to {output_filepath}...")
    df_clean.to_csv(output_filepath, index=False)
    
    # Print out summary statistics
    initial_size_mb: float = os.path.getsize(input_filepath) / (1024 * 1024)
    final_size_mb: float = os.path.getsize(output_filepath) / (1024 * 1024)
    
    print("\n--- Pipeline Success ---")
    print(f"Total Rows Processed: {len(df_clean)}")
    print(f"Raw File Size: {initial_size_mb:.2f} MB")
    print(f"Processed File Size: {final_size_mb:.2f} MB")
    print(f"Total Sessions: {df_clean['session_id'].nunique()}")

if __name__ == "__main__":
    # 1. Get the exact directory where this script lives
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build dynamic paths
    RAW_CSV_PATH: str = os.path.abspath(os.path.join(script_dir, "../../data/raw/lmcache_agentic_traces.csv"))
    PROCESSED_CSV_PATH: str = os.path.abspath(os.path.join(script_dir, "../../data/processed/processed_agentic_traces.csv"))
    
    process_agentic_traces(input_filepath=RAW_CSV_PATH, output_filepath=PROCESSED_CSV_PATH)