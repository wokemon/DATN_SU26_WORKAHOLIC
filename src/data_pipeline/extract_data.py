import os
import pandas as pd
from datasets import load_dataset, DatasetDict

def fetch_and_export_data(hf_repo_id: str, output_filepath: str) -> None:
    """
    Fetches a dataset from Hugging Face and exports the default/train split to a CSV file.
    
    Args:
        hf_repo_id (str): The Hugging Face repository ID.
        output_filepath (str): The local path where the CSV will be saved.
    """
    print(f"Starting extraction for dataset: {hf_repo_id}...")
    
    # We load the dataset directly from Hugging Face to ensure we get the latest upstream version.
    dataset: DatasetDict = load_dataset(hf_repo_id)
    
    # The dataset uses a 'train' split by default. 
    # We convert this split to a Pandas DataFrame because it natively supports 
    # highly optimized CSV writing and simplifies downstream flattening tasks.
    df: pd.DataFrame = dataset['train'].to_pandas()
    
    # Extract the directory path from the file path to ensure the target folder exists.
    # Why: Attempting to write a file to a non-existent directory raises a FileNotFoundError.
    output_dir: str = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Export to CSV without the index column to keep the file size manageable 
    # and avoid duplicate identifier columns in Power BI later.
    df.to_csv(output_filepath, index=False)
    
    print(f"Success! Exported {len(df)} rows to {output_filepath}")

if __name__ == "__main__":
    
    # 1. Get the exact directory where this script lives
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the path dynamically: go up two levels, then into data/raw/
    raw_path: str = os.path.join(script_dir, "../../data/raw/lmcache_agentic_traces.csv")
    
    # 3. Clean up the path format for your operating system
    OUTPUT_CSV_PATH: str = os.path.abspath(raw_path)
    
    DATASET_NAME: str = "sammshen/lmcache-agentic-traces"
    
    fetch_and_export_data(hf_repo_id=DATASET_NAME, output_filepath=OUTPUT_CSV_PATH)