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
            
            # Extract 'source' from 'session_id'
            if 'session_id' in self.df.columns:
                self.df['source'] = self.df['session_id'].str.split('__').str[0]
                logging.info("Successfully extracted 'source' from 'session_id'.")
            else:
                logging.warning("'session_id' column not found. Cannot extract 'source'.")
            
            logging.info(f"Data loaded successfully with {len(self.df)} records.")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def analyze_context_growth(self):
        """Visualizes how input tokens grow as the turn number increases, split into subplots by dataset source."""
        logging.info("Analyzing context growth and token usage by dataset (Horizontal Subplots)...")
        if not all(col in self.df.columns for col in ['turn_number', 'input_tokens', 'source']):
            logging.warning("Missing 'turn_number', 'input_tokens', or 'source' columns. Skipping context growth analysis.")
            return

        # --- UPDATED: Removed main legend, added facet_kws for independent axes ---
        g = sns.relplot(
            data=self.df, 
            x='turn_number', 
            y='input_tokens', 
            col='source',       
            hue='source',       
            kind='line', 
            errorbar='sd',
            height=5,           
            aspect=1.2,
            facet_kws={'sharey': False, 'sharex': False}, 
            legend=False  # Removes the redundant main side legend
        )
        
        # Customize overall figure layout and titles
        g.fig.suptitle('Context Explosion: Input Tokens vs. Turn Number', y=1.05, fontsize=16)
        g.set_titles(col_template="{col_name}")
        
        # --- NEW ADDITION: Iterate through each subplot to add specific Mean/Median lines ---
        for ax, source_name in zip(g.axes.flat, g.col_names):
            ax.set_xlabel('Turn Number')
            ax.set_ylabel('Input Tokens')
            
            # Filter the dataframe for the specific subplot's source
            source_data = self.df[self.df['source'] == source_name]['input_tokens']
            
            # Calculate metrics
            mean_tokens = source_data.mean()
            median_tokens = source_data.median()
            
            # Draw horizontal lines for mean and median
            ax.axhline(mean_tokens, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_tokens:.0f}')
            ax.axhline(median_tokens, color='purple', linestyle=':', linewidth=2, label=f'Median: {median_tokens:.0f}')
            
            # Add a small local legend inside each subplot just for the lines
            ax.legend(loc='upper left', fontsize=10)
        # ---------------------------------------------------------------------------------
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, 'context_growth.png'), bbox_inches='tight')
        plt.close()
        
    def analyze_token_distribution(self):
        """Analyzes the distribution of input vs. output tokens."""
        logging.info("Analyzing input vs. output token distribution...")
        if not all(col in self.df.columns for col in ['input_tokens', 'output_length']):
            logging.warning("Missing 'input_tokens' or 'output_length' columns. Skipping token distribution.")
            return

        # Melting the dataframe for side-by-side seaborn plotting
        token_df = self.df[['input_tokens', 'output_length']].melt(
            var_name='Token Type', value_name='Token Count'
        )

        plt.figure(figsize=(10, 6))
        # Log scale boxplot to better visualize distributions without massive outliers warping the view
        sns.boxplot(data=token_df, x='Token Type', y='Token Count', palette='Set2')
        plt.yscale('log')
        plt.title('Distribution: Input vs. Output Tokens (Log Scale)')
        plt.ylabel('Token Count (Log Scale)')
        plt.xlabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'token_distribution.png'))
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
        
        # Calculate Mean and Median
        mean_turns = session_stats['total_turns'].mean()
        median_turns = session_stats['total_turns'].median()
        
        # Add vertical lines for mean and median
        plt.axvline(mean_turns, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_turns:.1f}')
        plt.axvline(median_turns, color='purple', linestyle='-', linewidth=2, label=f'Median: {median_turns:.1f}')
        
        plt.title('Agent Efficiency: Total Turns Required per Session')
        plt.xlabel('Total Turns')
        plt.ylabel('Number of Sessions')
        plt.legend()
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
        self.analyze_token_distribution()
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