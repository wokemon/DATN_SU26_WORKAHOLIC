import os
import re
import pandas as pd

# Define approximate pricing per 1 million tokens (Input, Output) in USD
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

def extract_tool_called(text: str) -> str:
    """Extracts the tool name from the OpenHands <function=...> tag."""
    if not isinstance(text, str):
        return "none"
    match = re.search(r'<function=([^>]+)>', text)
    return match.group(1) if match else "none"

def check_for_errors(text: str) -> int:
    """Scans for common execution failure keywords."""
    if not isinstance(text, str):
        return 0
    error_keywords = ['Traceback', 'ModuleNotFoundError', 'Exit 1', 'Error:']
    return 1 if any(keyword in text for keyword in error_keywords) else 0

def process_agentic_traces(input_filepath: str, output_filepath: str, chunk_size: int = 2000) -> None:
    """
    Loads raw telemetry data safely using chunks, extracts performance/cost features, 
    and exports a lightweight CSV for PowerBI/ML models.
    """
    print(f"Starting chunked processing of {input_filepath}...")
    
    lightweight_chunks = []
    
    # 1. Process in Chunks to avoid RAM exhaustion
    for chunk_idx, chunk in enumerate(pd.read_csv(input_filepath, chunksize=chunk_size)):
        print(f"Processing Chunk {chunk_idx + 1}...")
        
        # --- NEW EXTRACTION LOGIC ---
        # Extract tool_called
        chunk['tool_called'] = chunk['input'].apply(extract_tool_called)
        
        # Extract has_error flag
        chunk['has_error'] = chunk['input'].apply(check_for_errors)
        
        # Flag System Prompt (Usually only on Turn 1)
        chunk['is_system_prompt_present'] = chunk['input'].apply(
            lambda x: 1 if isinstance(x, str) and '<ROLE>' in x else 0
        )
        
        # Estimate input tokens (Heuristic to save memory/compute)
        chunk['input_tokens'] = (chunk['input'].astype(str).str.len() // 4).astype(int)
        
        # Drop the massive 'input' column IMMEDIATELY to free up RAM
        chunk_clean = chunk.drop(columns=['input'])
        lightweight_chunks.append(chunk_clean)

    print("All chunks processed! Assembling the lightweight dataset...")
    
    # 2. Reassemble the lightweight dataset
    df_final: pd.DataFrame = pd.concat(lightweight_chunks, ignore_index=True)
    
    # 3. Sequence Mapping (Must be done after concat so session_ids aren't broken across chunks)
    df_final['turn_number'] = df_final.groupby('session_id').cumcount() + 1
    
    # 4. Cost Calculation
    df_final['turn_cost'] = df_final.apply(
        lambda row: calculate_turn_cost(
            model=row['model'], 
            input_tokens=row['input_tokens'], 
            output_tokens=row['output_length']
        ), 
        axis=1
    )
    
    # Ensure output directory exists
    output_dir: str = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Export the processed data
    print(f"Saving final dataset to {output_filepath}...")
    df_final.to_csv(output_filepath, index=False)
    
    # Print out summary statistics
    initial_size_mb: float = os.path.getsize(input_filepath) / (1024 * 1024)
    final_size_mb: float = os.path.getsize(output_filepath) / (1024 * 1024)
    
    print("\n--- Pipeline Success ---")
    print(f"Total Rows Processed: {len(df_final)}")
    print(f"Raw File Size: {initial_size_mb:.2f} MB")
    print(f"Processed File Size: {final_size_mb:.2f} MB")
    print(f"Total Sessions: {df_final['session_id'].nunique()}")

if __name__ == "__main__":
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    RAW_CSV_PATH: str = os.path.abspath(os.path.join(script_dir, "../../data/raw/lmcache_agentic_traces.csv"))
    PROCESSED_CSV_PATH: str = os.path.abspath(os.path.join(script_dir, "../../data/processed/processed_agentic_traces.csv"))
    
    process_agentic_traces(input_filepath=RAW_CSV_PATH, output_filepath=PROCESSED_CSV_PATH)