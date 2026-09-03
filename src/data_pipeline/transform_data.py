import os
import sys
import pandas as pd

# Allow running this file directly (e.g. `python src/data_pipeline/transform_data.py`)
# by putting the repo root on sys.path so the `src` package resolves.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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

    FIX (vs. original transform_data.py): `input` is the CUMULATIVE conversation
    history (input[N] is a strict prefix-superset of input[N-1], per the HF dataset
    card). Running check_for_errors() on the full cumulative `input` means that once
    an error keyword appears anywhere in a session's history, EVERY subsequent turn
    is also flagged as has_error=1 forever — even turns where the agent already
    recovered and ran cleanly. This produced a monotonically non-decreasing has_error
    signal for 99.87% of sessions (766/767) in the original output, and 100% of
    Minimax/DeepSeek sessions were flagged as "error" from turn 1 itself, before the
    agent had taken any action — almost certainly because the SWE-bench issue
    description/system prompt text itself contains one of the trigger keywords
    (e.g. a traceback quoted as part of the bug report), not because the agent failed.

    This version instead computes has_error on the INCREMENTAL new text added at each
    turn (the string diff between this turn's cumulative `input` and the previous
    turn's `input`, within the same session) — i.e. only what's actually new at this
    step: the model's new response and/or the new tool result it received.
    """
    print(f"Starting chunked processing of {input_filepath}...")

    lightweight_chunks = []
    raw_chunks = []  # keep 'input' + 'session_id' temporarily for the incremental diff

    # 1. Process in Chunks to avoid RAM exhaustion
    for chunk_idx, chunk in enumerate(pd.read_csv(input_filepath, chunksize=chunk_size)):
        print(f"Processing Chunk {chunk_idx + 1}...")

        # --- EXTRACTION LOGIC (non-error-dependent parts, same as before) ---
        chunk['is_system_prompt_present'] = chunk['input'].apply(
            lambda x: 1 if isinstance(x, str) and '<ROLE>' in x else 0
        )
        chunk['input_tokens'] = (chunk['input'].astype(str).str.len() // 4).astype(int)

        # Keep 'input' for now — needed for the incremental has_error diff below.
        # We drop it only after has_error has been computed on the FULL dataset
        # (the diff needs turns to be sorted globally within each session, which
        # chunk-local processing cannot guarantee across chunk boundaries).
        raw_chunks.append(chunk[['session_id', 'input']].copy())
        lightweight_chunks.append(chunk.drop(columns=['input']))

    print("All chunks processed! Assembling the lightweight dataset...")

    # 2. Reassemble the lightweight dataset
    df_final: pd.DataFrame = pd.concat(lightweight_chunks, ignore_index=True)
    df_raw_input: pd.DataFrame = pd.concat(raw_chunks, ignore_index=True)
    df_final['input'] = df_raw_input['input']

    # 3. Sequence Mapping
    # turn_number relies on rows for a session appearing in chronological order in the
    # source file. Capture the original row order explicitly and sort by it before the
    # cumcount so this invariant holds even if chunks are ever produced/merged out of order.
    df_final['_source_row_order'] = df_final.index
    df_final = df_final.sort_values(['session_id', '_source_row_order']).reset_index(drop=True)
    df_final['turn_number'] = df_final.groupby('session_id').cumcount() + 1

    # 4. FIXED has_error: check only the NEW text added at each turn, not the full
    # cumulative history. Since input[N] is a strict prefix-superset of input[N-1],
    # the incremental new content is simply the string suffix added since the
    # previous turn in the same session. Turn 1 has no previous turn, so its
    # increment is its full (short) input — system prompt + initial task only,
    # which is expected to rarely trigger an error keyword on its own.
    import ast

    def parse_messages(raw: str):
        """input is a numpy-array repr of dicts after the CSV round-trip (elements
        joined by '\\n ' inside brackets, not by ','), so it is NOT valid
        Python/JSON literal syntax as-is. Normalize separators before parsing."""
        if not isinstance(raw, str) or not raw.strip():
            return []
        text = raw.strip()
        if text.startswith('[') and text.endswith(']'):
            inner = text[1:-1]
            # numpy object-array repr separates elements with whitespace/newlines,
            # not commas -> insert ',' between adjacent '}...{' element boundaries.
            inner = inner.replace('}\n {', '}, {').replace('} {', '}, {')
            text = '[' + inner + ']'
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []

    def incremental_new_messages(group: pd.Series) -> pd.Series:
        parsed = group.apply(parse_messages)
        out = []
        prev_len = 0
        for msgs in parsed:
            new_msgs = msgs[prev_len:] if len(msgs) >= prev_len else msgs
            out.append(" ".join(str(m.get('content', '')) for m in new_msgs if isinstance(m, dict)))
            prev_len = len(msgs)
        return pd.Series(out, index=group.index)

    df_final['_input_increment'] = df_final.groupby('session_id')['input'].transform(incremental_new_messages)
    df_final['has_error'] = df_final['_input_increment'].apply(check_for_errors)

    # Now safe to drop the heavy raw text columns.
    df_final = df_final.drop(columns=['input', '_input_increment', '_source_row_order'])

    # 5. Cost Calculation (Using Modularized Feature)
    print("Applying token cost calculations via token_calculator module...")
    calculator = TokenCalculator()

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
    print(f"has_error rate (sanity check, should be well below the old ~90%): {df_final['has_error'].mean()*100:.1f}%")


if __name__ == "__main__":
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    RAW_CSV_PATH: str = os.path.abspath(os.path.join(script_dir, "../../data/raw/lmcache_agentic_traces.csv"))
    PROCESSED_CSV_PATH: str = os.path.abspath(os.path.join(script_dir, "../../data/processed/processed_agentic_traces_FIXED.csv"))

    process_agentic_traces(input_filepath=RAW_CSV_PATH, output_filepath=PROCESSED_CSV_PATH)