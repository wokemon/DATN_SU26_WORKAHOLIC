import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TelemetryEDA:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Styling for plots
        sns.set_theme(style="whitegrid")

    def load_data(self):
        """Loads the processed telemetry data."""
        logging.info(f"Loading data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully with {len(self.df)} records.")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def analyze_context_growth(self):
        """Visualizes how input tokens grow as the turn number increases."""
        logging.info("Analyzing context growth and token usage...")
        if 'turn_number' not in self.df.columns or 'input_tokens' not in self.df.columns:
            logging.warning("Missing 'turn_number' or 'input_tokens' columns. Skipping context growth analysis.")
            return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.df, x='turn_number', y='input_tokens', errorbar='sd')
        plt.title('Context Explosion: Input Tokens vs. Turn Number')
        plt.xlabel('Turn Number')
        plt.ylabel('Input Tokens')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'context_growth.png'))
        plt.close()
        
    def analyze_latency_bottlenecks(self):
        """Investigates latency (pre_gap) distributions using logarithmic scaling."""
        logging.info("Analyzing latency and performance bottlenecks (Log Scaled)...")
        if 'pre_gap' not in self.df.columns:
            logging.warning("Missing 'pre_gap' column. Skipping latency analysis.")
            return

        # Drop NaNs and ensure strictly positive values for log scaling
        latency_data = self.df['pre_gap'].dropna()
        latency_data = latency_data[latency_data > 0]

        plt.figure(figsize=(10, 6))
        
        # Use log_scale=True on the x-axis to reveal the true distribution of the long tail
        sns.histplot(latency_data, bins=50, kde=False, color='coral', log_scale=True)
        
        plt.title('Distribution of Turn Latency (Logarithmic Scale)')
        plt.xlabel('Latency / Pre-gap (Seconds) - Log Scale')
        plt.ylabel('Frequency')
        
        # Mark 95th and 99th percentiles to pinpoint exact anomaly thresholds
        percentile_95 = latency_data.quantile(0.95)
        percentile_99 = latency_data.quantile(0.99)
        
        plt.axvline(percentile_95, color='red', linestyle='--', label=f'95th Percentile ({percentile_95:.2f}s)')
        plt.axvline(percentile_99, color='darkred', linestyle='-', label=f'99th Percentile ({percentile_99:.2f}s)')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'latency_distribution.png'))
        plt.close()

    def analyze_agent_efficiency(self):
        """Calculates average turns per session and total session costs."""
        logging.info("Analyzing agent efficiency and total costs...")
        required_cols = ['session_id', 'turn_number']
        if not all(col in self.df.columns for col in required_cols):
            logging.warning("Missing necessary columns for efficiency analysis.")
            return

        # Max turns per session (Task length)
        session_stats = self.df.groupby('session_id').agg(
            total_turns=('turn_number', 'max')
        ).reset_index()

        if 'turn_cost' in self.df.columns:
            cost_stats = self.df.groupby('session_id').agg(
                total_cost=('turn_cost', 'sum')
            ).reset_index()
            session_stats = pd.merge(session_stats, cost_stats, on='session_id')
            
            # Print high-level cost metrics
            avg_cost = session_stats['total_cost'].mean()
            max_cost = session_stats['total_cost'].max()
            logging.info(f"Average cost per session: ${avg_cost:.4f}")
            logging.info(f"Max cost for a single session: ${max_cost:.4f}")

        # Plot distribution of total turns
        plt.figure(figsize=(10, 6))
        sns.histplot(session_stats['total_turns'], bins=20, kde=True, color='teal')
        plt.title('Agent Efficiency: Total Turns Required per Session')
        plt.xlabel('Total Turns')
        plt.ylabel('Number of Sessions')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'turn_efficiency_distribution.png'))
        plt.close()

        # Flag stuck sessions (e.g., hit 50 turns)
        stuck_sessions = session_stats[session_stats['total_turns'] >= 50]
        logging.info(f"Identified {len(stuck_sessions)} sessions that hit the 50-turn cap (likely stuck loops).")

    def run_pipeline(self):
        """Executes the full EDA pipeline."""
        self.load_data()
        self.analyze_context_growth()
        self.analyze_latency_bottlenecks()
        self.analyze_agent_efficiency()
        logging.info(f"EDA Complete. All plots saved to {self.output_dir}")

if __name__ == "__main__":
    # Define paths relative to the script's execution context
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH: str = os.path.abspath(os.path.join(script_dir, "../../data/processed/processed_agentic_traces.csv"))
    OUTPUT_DIR: str = os.path.abspath(os.path.join(script_dir, "../../data/exploration_outputs/"))

    eda = TelemetryEDA(data_path=DATA_PATH, output_dir=OUTPUT_DIR)
    eda.run_pipeline()