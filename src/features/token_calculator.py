import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pricing per 1,000,000 tokens (USD) configured exactly for your dataset
MODEL_PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "minimax-m2.5": {"input": 0.15, "output": 1.15},
    "deepseek-v3.1": {"input": 0.21, "output": 0.28},
    "default": {"input": 1.00, "output": 2.00}
}

class TokenCalculator:
    def __init__(self, pricing_table: dict = None):
        """Initializes the calculator with the dataset-specific pricing table."""
        self.pricing_table = pricing_table or MODEL_PRICING

    def normalize_model_name(self, raw_model_name: str) -> str:
        """Fuzzy matches raw string data to the exact keys in MODEL_PRICING."""
        if not isinstance(raw_model_name, str):
            return "default"
            
        name_lower = raw_model_name.lower()
        
        if "sonnet" in name_lower or "claude" in name_lower:
            return "claude-sonnet-4-6"
        elif "opus" in name_lower:
            return "claude-opus-4-6"
        elif "minimax" in name_lower:
            return "minimax-m2.5"
        elif "deepseek" in name_lower:
            return "deepseek-v3.1"
            
        return "default"

    def calculate_single_turn(self, raw_model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculates the USD cost for a single agentic turn."""
        in_toks = max(0, int(input_tokens)) if pd.notnull(input_tokens) else 0
        out_toks = max(0, int(output_tokens)) if pd.notnull(output_tokens) else 0

        model_key = self.normalize_model_name(raw_model_name)
        rates = self.pricing_table.get(model_key, self.pricing_table["default"])

        input_cost = (in_toks / 1_000_000) * rates["input"]
        output_cost = (out_toks / 1_000_000) * rates["output"]

        return input_cost + output_cost

    def apply_costs_to_dataframe(self, df: pd.DataFrame, model_col: str = 'model', 
                                 in_col: str = 'input_tokens', out_col: str = 'output_tokens') -> pd.DataFrame:
        """Vectorized cost calculation across the DataFrame."""
        logging.info("Applying financial cost calculations to token data...")
        
        missing_cols = [col for col in [model_col, in_col, out_col] if col not in df.columns]
        if missing_cols:
            logging.error(f"Cannot calculate costs. Missing columns: {missing_cols}")
            raise KeyError(f"Missing columns: {missing_cols}")

        df['turn_cost'] = df.apply(
            lambda row: self.calculate_single_turn(
                row[model_col], 
                row[in_col], 
                row[out_col]
            ), axis=1
        )
        
        logging.info(f"Cost calculation complete. Total dataset cost: ${df['turn_cost'].sum():.4f}")
        return df