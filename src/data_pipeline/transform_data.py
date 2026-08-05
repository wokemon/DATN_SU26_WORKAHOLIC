import os
import pandas as pd

# Import the feature module we just created
from src.features.token_calculator import TokenCalculator

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
        
        # --- EXTRACTION LOGIC ---
        chunk['has_error'] = chunk['input'].apply(check_for_errors)
        chunk['is_system_prompt_present'] = chunk['input'].apply(
            lambda x: 1 if isinstance(x, str) and '<ROLE>' in x else 0
        )
        chunk['input_tokens'] = (chunk['input'].astype(str).str.len() // 4).astype(int)
        
        # Drop the massive 'input' column IMMEDIATELY to free up RAM
        chunk_clean = chunk.drop(columns=['input'])
        lightweight_chunks.append(chunk_clean)

    print("All chunks processed! Assembling the lightweight dataset...")
    
    # 2. Reassemble the lightweight dataset
    df_final: pd.DataFrame = pd.concat(lightweight_chunks, ignore_index=True)

    # 3. Sequence Mapping
    # turn_number relies on rows for a session appearing in chronological order in the
    # source file. Capture the original row order explicitly and sort by it before the
    # cumcount so this invariant holds even if chunks are ever produced/merged out of order.
    df_final['_source_row_order'] = df_final.index
    df_final = df_final.sort_values(['session_id', '_source_row_order']).reset_index(drop=True)
    df_final['turn_number'] = df_final.groupby('session_id').cumcount() + 1
    df_final = df_final.drop(columns=['_source_row_order'])
    
    # 4. Cost Calculation (Using Modularized Feature)
    print("Applying token cost calculations via token_calculator module...")
    calculator = TokenCalculator()
    
    # Apply calculation (Mapping out_col to 'output_length' based on your dataset)
    df_final = calculator.apply_costs_to_dataframe(
        df=df_final, 
        model_col='model', 
        in_col='input_tokens', 
        out_col='output_length'  
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