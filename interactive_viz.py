"""
Interactive visualization module for Clinical Valence Testing.

This module provides interactive Plotly-based visualizations for exploring
valence testing results.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
import logging

logger = logging.getLogger(__name__)


class InteractiveVisualizer:
    """
    Creates interactive visualizations for valence testing results.

    Features:
    - Interactive heatmaps with hover information
    - 3D surface plots of diagnosis shifts
    - Time-series plots of attention weights
    - Distribution comparisons with statistical annotations
    - Exportable HTML dashboards
    """

    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize the interactive visualizer.

        Args:
            results_dir: Directory containing result CSV files
        """
        self.results_dir = Path(results_dir)
        self.diagnosis_dfs = {}
        self.attention_dfs = {}
        self.notes_dfs = {}
        self._load_results()

    def _load_results(self):
        """Load all results files from the directory."""
        logger.info(f"Loading results from {self.results_dir}")

        for file in self.results_dir.glob("*_diagnosis.csv"):
            name = file.stem.split('_')[0]
            self.diagnosis_dfs[name] = pd.read_csv(file)
            logger.info(f"Loaded diagnosis results for {name}")

        for file in self.results_dir.glob("*_attention.csv"):
            name = file.stem.split('_')[0]
            self.attention_dfs[name] = pd.read_csv(file)
            logger.info(f"Loaded attention results for {name}")

        for file in self.results_dir.glob("*_clinical_notes.csv"):
            name = file.stem.split('_')[0]
            self.notes_dfs[name] = pd.read_csv(file)
            logger.info(f"Loaded clinical notes for {name}")

    def create_interactive_heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        colorscale: str = "RdBu_r",
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create an interactive heatmap with hover information.

        Args:
            data: DataFrame with diagnosis codes as columns and valence as rows
            title: Title for the plot
            colorscale: Plotly colorscale name
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=colorscale,
            hovertemplate='<b>%{y}</b><br>' +
                         'Diagnosis: %{x}<br>' +
                         'Shift: %{z:.4f}<extra></extra>',
            colorbar=dict(title="Probability Shift")
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title="Diagnosis Code",
            yaxis_title="Valence Type",
            hovermode='closest',
            height=600,
            width=1000,
            template="plotly_white"
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive heatmap to {save_path}")

        return fig

    def create_diagnosis_comparison(
        self,
        baseline_name: str = "neutralize",
        treatment_names: Optional[List[str]] = None,
        diagnosis_codes: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create interactive comparison of diagnosis probabilities across valence types.

        Args:
            baseline_name: Name of baseline condition
            treatment_names: Names of treatment conditions to compare
            diagnosis_codes: Specific diagnosis codes to include (None = all)
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        if baseline_name not in self.diagnosis_dfs:
            raise ValueError(f"Baseline '{baseline_name}' not found in results")

        baseline_df = self.diagnosis_dfs[baseline_name]

        if treatment_names is None:
            treatment_names = [name for name in self.diagnosis_dfs.keys() if name != baseline_name]

        if diagnosis_codes is None:
            diagnosis_codes = [col for col in baseline_df.columns
                             if col not in ['NoteID', 'Valence', 'Val_class']][:10]  # Top 10

        # Create subplots
        fig = make_subplots(
            rows=len(diagnosis_codes),
            cols=1,
            subplot_titles=[f"Diagnosis: {code}" for code in diagnosis_codes],
            vertical_spacing=0.05
        )

        for i, code in enumerate(diagnosis_codes, 1):
            # Baseline
            fig.add_trace(
                go.Box(
                    y=baseline_df[code],
                    name=f"{baseline_name} (baseline)",
                    boxmean='sd',
                    marker_color='lightblue'
                ),
                row=i, col=1
            )

            # Treatments
            colors = px.colors.qualitative.Set2
            for j, treatment_name in enumerate(treatment_names):
                if treatment_name in self.diagnosis_dfs:
                    fig.add_trace(
                        go.Box(
                            y=self.diagnosis_dfs[treatment_name][code],
                            name=treatment_name,
                            boxmean='sd',
                            marker_color=colors[j % len(colors)]
                        ),
                        row=i, col=1
                    )

        fig.update_layout(
            title="Interactive Diagnosis Probability Comparison",
            height=300 * len(diagnosis_codes),
            width=1200,
            showlegend=True,
            template="plotly_white"
        )

        fig.update_yaxes(title_text="Probability", row=len(diagnosis_codes), col=1)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved diagnosis comparison to {save_path}")

        return fig

    def create_attention_explorer(
        self,
        valence_types: Optional[List[str]] = None,
        top_n_words: int = 30,
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create interactive attention weight explorer.

        Args:
            valence_types: Valence types to include
            top_n_words: Number of top words to show
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        if not self.attention_dfs:
            raise ValueError("No attention data available")

        # Combine all attention data
        combined = pd.concat([
            df.assign(source=name) for name, df in self.attention_dfs.items()
        ], ignore_index=True)

        if valence_types:
            combined = combined[combined['Val_class'].isin(valence_types)]

        # Get top words by mean attention weight
        top_words = (combined.groupby('Word')['AttentionWeight']
                    .mean()
                    .nlargest(top_n_words)
                    .index.tolist())

        # Filter to top words
        plot_data = combined[combined['Word'].isin(top_words)]

        # Create violin plot
        fig = go.Figure()

        valence_classes = plot_data['Val_class'].unique()
        colors = px.colors.qualitative.Plotly

        for i, valence in enumerate(valence_classes):
            valence_data = plot_data[plot_data['Val_class'] == valence]

            fig.add_trace(go.Violin(
                x=valence_data['Word'],
                y=valence_data['AttentionWeight'],
                name=valence,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.6,
                hovertemplate='<b>%{x}</b><br>' +
                             f'Valence: {valence}<br>' +
                             'Attention: %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            title="Interactive Attention Weight Explorer",
            xaxis_title="Word",
            yaxis_title="Attention Weight",
            height=700,
            width=1400,
            violinmode='group',
            template="plotly_white",
            hovermode='closest'
        )

        fig.update_xaxes(tickangle=45)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved attention explorer to {save_path}")

        return fig

    def create_3d_surface(
        self,
        diagnosis_codes: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create 3D surface plot of diagnosis shifts across valence types.

        Args:
            diagnosis_codes: Diagnosis codes to include
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        if len(self.diagnosis_dfs) < 2:
            raise ValueError("Need at least 2 valence types for 3D plot")

        # Get baseline
        baseline_df = self.diagnosis_dfs.get('neutralize', list(self.diagnosis_dfs.values())[0])

        if diagnosis_codes is None:
            diagnosis_codes = [col for col in baseline_df.columns
                             if col not in ['NoteID', 'Valence', 'Val_class']][:20]

        # Calculate mean shifts
        shifts = []
        valence_names = []

        for name, df in self.diagnosis_dfs.items():
            if name == 'neutralize':
                continue
            mean_probs = df[diagnosis_codes].mean()
            baseline_mean = baseline_df[diagnosis_codes].mean()
            shift = mean_probs - baseline_mean
            shifts.append(shift.values)
            valence_names.append(name)

        shifts_array = np.array(shifts)

        fig = go.Figure(data=[go.Surface(
            z=shifts_array,
            x=list(range(len(diagnosis_codes))),
            y=list(range(len(valence_names))),
            colorscale='RdBu_r',
            hovertemplate='Diagnosis: %{x}<br>' +
                         'Valence: %{y}<br>' +
                         'Shift: %{z:.4f}<extra></extra>'
        )])

        fig.update_layout(
            title="3D Surface Plot of Diagnosis Probability Shifts",
            scene=dict(
                xaxis_title="Diagnosis Code Index",
                yaxis_title="Valence Type Index",
                zaxis_title="Probability Shift",
                xaxis=dict(ticktext=diagnosis_codes, tickvals=list(range(len(diagnosis_codes)))),
                yaxis=dict(ticktext=valence_names, tickvals=list(range(len(valence_names))))
            ),
            height=800,
            width=1200,
            template="plotly_white"
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved 3D surface plot to {save_path}")

        return fig

    def create_dashboard(self, output_path: Union[str, Path]):
        """
        Create a comprehensive interactive dashboard with all visualizations.

        Args:
            output_path: Path to save the HTML dashboard
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Creating comprehensive dashboard...")

        # Create individual plots
        figures = []

        # 1. Diagnosis comparison
        try:
            fig1 = self.create_diagnosis_comparison()
            figures.append(("Diagnosis Comparison", fig1))
        except Exception as e:
            logger.warning(f"Could not create diagnosis comparison: {e}")

        # 2. Attention explorer
        try:
            fig2 = self.create_attention_explorer()
            figures.append(("Attention Explorer", fig2))
        except Exception as e:
            logger.warning(f"Could not create attention explorer: {e}")

        # 3. 3D surface
        try:
            fig3 = self.create_3d_surface()
            figures.append(("3D Surface Plot", fig3))
        except Exception as e:
            logger.warning(f"Could not create 3D surface: {e}")

        # Combine into HTML dashboard
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical Valence Testing - Interactive Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                .plot-container {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Clinical Valence Testing - Interactive Dashboard</h1>
            <p style="text-align: center; color: #666;">
                Explore how valence words affect clinical model predictions
            </p>
        """

        for title, fig in figures:
            html_content += f'<div class="plot-container"><h2>{title}</h2>'
            html_content += fig.to_html(include_plotlyjs=False, div_id=title.replace(" ", "_"))
            html_content += '</div>'

        html_content += """
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Dashboard saved to {output_path}")
        logger.info(f"Open {output_path} in a web browser to view")


def main():
    """Example usage of the interactive visualizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Create interactive visualizations")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing result CSV files")
    parser.add_argument("--output", type=str, default="dashboard.html",
                       help="Output path for dashboard")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create visualizer
    viz = InteractiveVisualizer(args.results_dir)

    # Create dashboard
    viz.create_dashboard(args.output)

    print(f"\n✅ Dashboard created successfully!")
    print(f"📊 Open {args.output} in your web browser to explore results")


if __name__ == "__main__":
    main()
