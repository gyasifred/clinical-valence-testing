import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import fire


@dataclass
class ValenceVisualizationConfig:
    """Configuration for valence shift visualization settings"""
    figure_size: tuple = (15, 10)
    font_scale: int = 2
    fixed_range: bool = True
    vmin: float = -0.1 
    vmax: float = 0.1
    cmap: str = "vlag"   
    tick_font_size: int = 22
    baseline: str = "neutralize"  


class ValenceShiftAnalyzer:
    ICD_TRANSLATIONS = {
        "401": "Hypertension",
        "427": "Cardiac dysrhythmias",
        "276": "Disorders of fluid electrolyte",
        "272": "Disorders of lipoid metabolism",
        "250": "Diabetes",
        "414": "Chronic ischemic heart disease",
        "428": "Heart failure",
        "518": "Other diseases of lung",
        "285": "Unspecified anemias",
        "584": "Acute kidney failure",
        "599": "Urinary tract disorders",
        "V586": "Drug Use",
        "530": "Diseases of esophagus",
        "585": "Chronic kidney disease",
        "403": "Hypertensive chronic kidney disease"
    }

    def __init__(self, results_dir: str, config: Optional[ValenceVisualizationConfig] = None):
        self.results_dir = Path(results_dir)
        self.diagnosis_dfs = {}
        self.attention_dfs = {}
        self.notes_dfs = {}
        self._load_results()
        self.config = config or ValenceVisualizationConfig()
        sns.set_style("darkgrid")

        self.combined_attention = pd.concat(self.attention_dfs.values(), ignore_index=True)
        self.combined_diagnosis = pd.concat(self.diagnosis_dfs.values(), ignore_index=True)
        self.combined_notes = pd.concat(self.notes_dfs.values(), ignore_index=True)

    def _load_results(self):
        """Load all results files from the directory"""
        self._load_csv_files("_diagnosis.csv", self.diagnosis_dfs)
        self._load_csv_files("_attention.csv", self.attention_dfs)
        self._load_csv_files("_clinical_notes.csv", self.notes_dfs)

    def _load_csv_files(self, file_pattern: str, dfs_dict: Dict[str, pd.DataFrame]):
        """Helper function to load CSV files matching the pattern"""
        for file in self.results_dir.glob(f"*{file_pattern}"):
            name = file.stem.split('_')[0]
            df = pd.read_csv(file)
            
            if 'AttentionWeight' in df.columns:
                df['AttentionWeight'] = pd.to_numeric(df['AttentionWeight'], errors='coerce')
            
            categorical_cols = ['Valence', 'Val_class', 'Word']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            dfs_dict[name] = df

    def _calculate_shifts_from_baseline(self, df: pd.DataFrame, value_column: str, 
                                     group_column: str = 'Val_class') -> pd.DataFrame:
        """Calculate shifts from baseline for any metric"""
        # Calculate mean values for each diagnosis and valence class
        pivot_df = df.pivot_table(
            values=value_column,
            index=list(set(df.columns) - {value_column, group_column}),
            columns=group_column,
            aggfunc='mean'
        )
        
        # Calculate shifts from baseline
        baseline_values = pivot_df[self.config.baseline]
        shift_df = pivot_df.sub(baseline_values, axis=0)
        
        # Rename columns to indicate they are shifts
        shift_df.columns = [f"{col}_shift" for col in shift_df.columns]
        
        return shift_df

    def _calculate_valence_shifts(self) -> pd.DataFrame:
        """Calculate diagnosis probability shifts from baseline"""
        cols = ['Val_class'] + list(self.ICD_TRANSLATIONS.keys())
        diagnosis_df = self.combined_diagnosis[cols]
        
        # Melt the dataframe to long format for diagnosis probabilities
        melted_df = diagnosis_df.melt(
            id_vars=['Val_class'],
            var_name='Diagnosis',
            value_name='Probability'
        )
        
        # Calculate shifts and process
        shift_df = self._calculate_shifts_from_baseline(melted_df, 'Probability')
        shift_df = shift_df.loc[list(self.ICD_TRANSLATIONS.keys())]
        shift_df.index = [self.ICD_TRANSLATIONS[code] for code in shift_df.index]
        
        return shift_df


    def plot_shifts(self, data: pd.DataFrame, title: str, save_path: Optional[Path] = None,
                   figsize: Optional[Tuple[int, int]] = None) -> Dict:
        """Generic function to plot shift heatmaps"""
        fig, ax = plt.subplots(figsize=figsize or self.config.figure_size)
        sns.set(font_scale=self.config.font_scale)
        
        sns.heatmap(
            data,
            ax=ax,
            vmin=self.config.vmin if self.config.fixed_range else None,
            vmax=self.config.vmax if self.config.fixed_range else None,
            cmap=self.config.cmap,
            center=0
        )
        
        ax.set_yticklabels(ax.get_yticklabels(), size=self.config.tick_font_size)
        plt.title(f"{title} (Baseline: {self.config.baseline})", pad=20)
        fig.subplots_adjust(left=0.35)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()
        
        return {
            "num_items": len(data),
            "valence_groups": [col.replace('_shift', '') 
                             for col in data.columns 
                             if col != f"{self.config.baseline}_shift"],
            "max_shift": data.abs().max().max(),
            "mean_shift": data.abs().mean().mean(),
            "most_affected_item": data.abs().mean(axis=1).idxmax()
        }

    def plot_valence_shifts(self, save_path: Optional[Path] = None) -> Dict:
        """Plot diagnosis probability shifts from baseline"""
        shift_df = self._calculate_valence_shifts()
        return self.plot_shifts(shift_df, "Diagnosis Probability Shifts", save_path)

    

def main(results_dir: str = "result2", 
         output_dir: str = "output", 
         fixed_range: bool = True,
         vmin: float = -0.035,
         vmax: float = 0.035,
         baseline: str = "neutralize"):
    
    config = ValenceVisualizationConfig(
        fixed_range=fixed_range,
        vmin=vmin,
        vmax=vmax,
        baseline=baseline
    )
    
    analyzer = ValenceShiftAnalyzer(results_dir=results_dir, config=config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save valence shifts
    valence_save_path = output_dir / "valence_shifts.png"
    valence_results = analyzer.plot_valence_shifts(save_path=valence_save_path)
    
    # Save analysis results
    results = {
        "valence_analysis": valence_results,
    }
    pd.DataFrame(results).to_csv(output_dir / "analysis_summary.csv")

    print(f"Visualizations saved to: {output_dir}")
    print(f"Summary saved to: {output_dir / 'analysis_summary.csv'}")


if __name__ == "__main__":
    fire.Fire(main)