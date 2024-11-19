import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class ValenceVisualizationConfig:
    """Configuration for valence shift visualization settings"""
    figure_size: tuple = (15, 15)
    font_scale: int = 2
    fixed_range: bool = True
    vmin: float = -0.1
    vmax: float = 0.1
    cmap: str = "vlag"
    tick_font_size: int = 22
    baseline: str = "neutralize"

class CommonWordShiftAnalyzer:
    def __init__(self, results_dir: str, config: Optional[ValenceVisualizationConfig] = None):
        self.results_dir = Path(results_dir)
        self.attention_dfs = {}
        self.config = config or ValenceVisualizationConfig()
        self._load_attention_data()
        sns.set_style("darkgrid")

    def _load_attention_data(self):
        """Load all attention CSV files from the directory"""
        for file in self.results_dir.glob("*_attention.csv"):
            name = file.stem.split('_')[0]
            df = pd.read_csv(file)
            df['AttentionWeight'] = pd.to_numeric(df['AttentionWeight'], errors='coerce')
            if 'Word' in df.columns:
                df['Word'] = df['Word'].astype('category')
            if 'Val_class' in df.columns:
                df['Val_class'] = df['Val_class'].astype('category')
            self.attention_dfs[name] = df

    def get_common_words(self):
        """Find words that are common across all datasets"""
        word_sets = [set(df['Word'].unique()) for df in self.attention_dfs.values()]
        common_words = set.intersection(*word_sets)
        return common_words

    def calculate_shifts(self, top_n: int = 30):
        """Calculate shifts for common words and return top N by absolute shift"""
        common_words = self.get_common_words()
        
        # Combine all attention data
        combined_df = pd.concat(self.attention_dfs.values(), ignore_index=True)
        combined_df = combined_df[combined_df['Word'].isin(common_words)]
        
        # Calculate mean values for each word and valence class
        pivot_df = combined_df.pivot_table(
            values='AttentionWeight',
            index='Word',
            columns='Val_class',
            aggfunc='mean'
        )
        
        # Calculate shifts from baseline
        baseline_values = pivot_df[self.config.baseline]
        shift_df = pivot_df.sub(baseline_values, axis=0)
        shift_df.columns = [f"{col}_shift" for col in shift_df.columns]
        
        # Select top N words by absolute mean shift
        mean_shifts = shift_df.abs().mean(axis=1)
        top_words = mean_shifts.nlargest(top_n).index
        return shift_df.loc[top_words]

    def plot_top_shifts(self, save_path: Optional[Path] = None):
        """Plot shifts for top 30 common words"""
        shift_df = self.calculate_shifts(top_n=30)
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        sns.set(font_scale=self.config.font_scale)
        
        sns.heatmap(
            shift_df,
            ax=ax,
            vmin=self.config.vmin if self.config.fixed_range else None,
            vmax=self.config.vmax if self.config.fixed_range else None,
            cmap=self.config.cmap,
            center=0
        )
        
        ax.set_yticklabels(ax.get_yticklabels(), size=self.config.tick_font_size)
        plt.title(f"Attention Weight Shifts for Top 30 Common Words\n(Baseline: {self.config.baseline})", pad=20)
        fig.subplots_adjust(left=0.35)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()

        return {
            "num_words": len(shift_df),
            "max_shift": shift_df.abs().max().max(),
            "mean_shift": shift_df.abs().mean().mean(),
            "most_affected_word": shift_df.abs().mean(axis=1).idxmax()
        }

# Usage example
def main():
    config = ValenceVisualizationConfig()
    analyzer = CommonWordShiftAnalyzer(results_dir="result2", config=config)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save the results
    save_path = output_dir / "common_word_shifts.png"
    results = analyzer.plot_top_shifts(save_path=save_path)
    
    # Save analysis results
    pd.DataFrame([results]).to_csv(output_dir / "common_words_analysis.csv")
    
    print(f"Visualization saved to: {save_path}")
    print(f"Analysis summary saved to: {output_dir / 'common_words_analysis.csv'}")

if __name__ == "__main__":
    main()