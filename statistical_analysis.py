"""
Statistical Analysis Module for Clinical Valence Testing.

This module provides comprehensive statistical analysis functions for
evaluating the effect of valence shifts on clinical predictions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'significant': self.significant,
            'effect_size': self.effect_size,
            'ci_lower': self.confidence_interval[0] if self.confidence_interval else None,
            'ci_upper': self.confidence_interval[1] if self.confidence_interval else None,
            **(self.additional_info or {})
        }


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for valence testing results.

    This class provides methods for:
    - Hypothesis testing (paired t-tests, Wilcoxon, etc.)
    - Effect size calculation (Cohen's d, Hedges' g, etc.)
    - Confidence interval estimation
    - Multiple comparison correction
    - Distribution analysis
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        correction_method: str = "fdr_bh"
    ):
        """
        Initialize statistical analyzer.

        Args:
            significance_level: Alpha level for significance testing
            correction_method: Method for multiple comparison correction
                             ('bonferroni', 'fdr_bh', 'fdr_by', or 'none')
        """
        self.significance_level = significance_level
        self.correction_method = correction_method
        logger.info(
            f"Initialized StatisticalAnalyzer with alpha={significance_level}, "
            f"correction={correction_method}"
        )

    def cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = True
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group of observations
            group2: Second group of observations
            paired: Whether groups are paired

        Returns:
            Cohen's d effect size (returns 0.0 if std is zero)
        """
        if paired:
            diff = group1 - group2
            std_diff = np.std(diff, ddof=1)
            if std_diff == 0 or np.isnan(std_diff):
                return 0.0  # No variance, effect size is zero
            return np.mean(diff) / std_diff
        else:
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std == 0 or np.isnan(pooled_std):
                return 0.0  # No variance, effect size is zero
            return (np.mean(group1) - np.mean(group2)) / pooled_std

    def hedges_g(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = True
    ) -> float:
        """
        Calculate Hedges' g effect size (bias-corrected Cohen's d).

        Args:
            group1: First group of observations
            group2: Second group of observations
            paired: Whether groups are paired

        Returns:
            Hedges' g effect size
        """
        d = self.cohens_d(group1, group2, paired)
        n = len(group1) if paired else len(group1) + len(group2)
        correction_factor = 1 - (3 / (4 * n - 9))
        return d * correction_factor

    def paired_ttest(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """
        Perform paired t-test.

        Args:
            baseline: Baseline group observations
            treatment: Treatment group observations
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

        Returns:
            Statistical test result
        """
        statistic, p_value = ttest_rel(baseline, treatment, alternative=alternative)
        effect_size = self.cohens_d(treatment, baseline, paired=True)

        # Calculate confidence interval for mean difference
        diff = treatment - baseline
        se = stats.sem(diff)
        ci = stats.t.interval(
            1 - self.significance_level,
            len(diff) - 1,
            loc=np.mean(diff),
            scale=se
        )

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=effect_size,
            confidence_interval=ci,
            additional_info={
                'mean_difference': np.mean(diff),
                'std_difference': np.std(diff, ddof=1),
                'alternative': alternative
            }
        )

    def wilcoxon_test(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric paired test).

        Args:
            baseline: Baseline group observations
            treatment: Treatment group observations
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

        Returns:
            Statistical test result
        """
        statistic, p_value = wilcoxon(
            baseline,
            treatment,
            alternative=alternative,
            zero_method='wilcox'
        )

        # Calculate rank-biserial correlation as effect size
        diff = treatment - baseline
        n_pos = np.sum(diff > 0)
        n_neg = np.sum(diff < 0)
        r = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0

        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=r,
            additional_info={
                'median_difference': np.median(diff),
                'n_positive': int(n_pos),
                'n_negative': int(n_neg),
                'alternative': alternative
            }
        )

    def mann_whitney_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """
        Perform Mann-Whitney U test (non-parametric unpaired test).

        Args:
            group1: First group observations
            group2: Second group observations
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

        Returns:
            Statistical test result
        """
        statistic, p_value = mannwhitneyu(
            group1,
            group2,
            alternative=alternative
        )

        # Calculate rank-biserial correlation
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * statistic) / (n1 * n2)

        return StatisticalTestResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=r,
            additional_info={
                'median_group1': np.median(group1),
                'median_group2': np.median(group2),
                'alternative': alternative
            }
        )

    def paired_permutation_test(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        n_permutations: int = 10000,
        alternative: str = "two-sided",
        random_seed: Optional[int] = None
    ) -> StatisticalTestResult:
        """
        Perform paired permutation test (approximate randomization).

        This is a non-parametric test that doesn't assume normality.
        It tests whether the observed difference is significant by
        randomly permuting the assignment of baseline/treatment labels.

        Reference: Yeh (2000) - More Accurate Tests for the Statistical
        Significance of Result Differences.

        Args:
            baseline: Baseline group observations
            treatment: Treatment group observations
            n_permutations: Number of random permutations
            alternative: Alternative hypothesis ('two-sided', 'greater', 'less')
            random_seed: Random seed for reproducibility

        Returns:
            StatisticalTestResult with permutation p-value
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Calculate observed difference
        observed_diff = np.mean(treatment - baseline)

        # Generate permutations
        n_samples = len(baseline)
        count_extreme = 0

        # Vectorized approach for efficiency
        flip_masks = np.random.choice([-1, 1], size=(n_permutations, n_samples))
        diffs = treatment - baseline
        permuted_diffs = flip_masks * diffs[np.newaxis, :]
        permuted_means = np.mean(permuted_diffs, axis=1)

        # Count extreme values based on alternative hypothesis
        if alternative == "two-sided":
            count_extreme = np.sum(np.abs(permuted_means) >= abs(observed_diff))
        elif alternative == "greater":
            count_extreme = np.sum(permuted_means >= observed_diff)
        elif alternative == "less":
            count_extreme = np.sum(permuted_means <= observed_diff)

        # Calculate p-value
        p_value = (count_extreme + 1) / (n_permutations + 1)

        # Calculate effect size (same as paired t-test)
        effect_size = self.cohens_d(treatment, baseline, paired=True)

        # Bootstrap CI for mean difference
        _, ci = self.bootstrap_confidence_interval(
            treatment - baseline,
            np.mean,
            n_bootstrap=5000
        )

        return StatisticalTestResult(
            test_name="Paired Permutation Test",
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=effect_size,
            confidence_interval=ci,
            additional_info={
                'n_permutations': n_permutations,
                'alternative': alternative,
                'mean_difference': observed_diff,
                'method': 'approximate_randomization'
            }
        )

    def stratified_permutation_test(
        self,
        baseline_binary: np.ndarray,
        treatment_binary: np.ndarray,
        n_permutations: int = 10000,
        random_seed: Optional[int] = None
    ) -> StatisticalTestResult:
        """
        Stratified permutation test for binary classification outcomes.

        Stratifies by agreement pattern:
        - Both positive (TP-TP): Keep fixed
        - Both negative (TN-TN): Keep fixed
        - Baseline+/Treatment- (TP-FN): Randomize
        - Baseline-/Treatment+ (FN-TP): Randomize

        This preserves the comparison structure while testing significance,
        following Yeh (2000)'s stratified randomization approach.

        Args:
            baseline_binary: Binary predictions for baseline (0 or 1)
            treatment_binary: Binary predictions for treatment (0 or 1)
            n_permutations: Number of permutations
            random_seed: Random seed

        Returns:
            StatisticalTestResult with accuracy difference and p-value
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Identify strata
        both_positive = (baseline_binary == 1) & (treatment_binary == 1)
        both_negative = (baseline_binary == 0) & (treatment_binary == 0)
        baseline_only = (baseline_binary == 1) & (treatment_binary == 0)
        treatment_only = (baseline_binary == 0) & (treatment_binary == 1)

        # Calculate observed accuracy difference
        observed_metric = np.mean(treatment_binary) - np.mean(baseline_binary)

        count_extreme = 0
        n_disagree = np.sum(baseline_only) + np.sum(treatment_only)

        if n_disagree == 0:
            # No disagreements - perfect agreement
            return StatisticalTestResult(
                test_name="Stratified Permutation Test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                additional_info={
                    'n_permutations': n_permutations,
                    'n_both_positive': int(np.sum(both_positive)),
                    'n_both_negative': int(np.sum(both_negative)),
                    'n_disagreements': 0,
                    'note': 'Perfect agreement - no permutation needed'
                }
            )

        disagree_indices = np.where(baseline_only | treatment_only)[0]
        n_baseline_only = np.sum(baseline_only)

        for _ in range(n_permutations):
            # Create permuted arrays
            perm_baseline = baseline_binary.copy()
            perm_treatment = treatment_binary.copy()

            # Randomly reassign disagreements
            random_assignment = np.random.permutation(n_disagree)
            new_baseline_only_idx = disagree_indices[random_assignment < n_baseline_only]
            new_treatment_only_idx = disagree_indices[random_assignment >= n_baseline_only]

            # Reset disagreement positions
            perm_baseline[disagree_indices] = 0
            perm_treatment[disagree_indices] = 0

            # Set new assignments
            perm_baseline[new_baseline_only_idx] = 1
            perm_treatment[new_treatment_only_idx] = 1

            # Preserve agreement strata
            perm_baseline[both_positive] = 1
            perm_treatment[both_positive] = 1
            perm_baseline[both_negative] = 0
            perm_treatment[both_negative] = 0

            # Calculate permuted metric
            permuted_metric = np.mean(perm_treatment) - np.mean(perm_baseline)

            if abs(permuted_metric) >= abs(observed_metric):
                count_extreme += 1

        p_value = (count_extreme + 1) / (n_permutations + 1)

        return StatisticalTestResult(
            test_name="Stratified Permutation Test",
            statistic=observed_metric,
            p_value=p_value,
            significant=p_value < self.significance_level,
            additional_info={
                'n_permutations': n_permutations,
                'n_both_positive': int(np.sum(both_positive)),
                'n_both_negative': int(np.sum(both_negative)),
                'n_baseline_only': int(np.sum(baseline_only)),
                'n_treatment_only': int(np.sum(treatment_only)),
                'method': 'stratified_randomization'
            }
        )

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate bootstrap confidence interval for a statistic.

        Args:
            data: Input data
            statistic_func: Function to calculate statistic (e.g., np.mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for interval

        Returns:
            Tuple of (point estimate, confidence interval)
        """
        bootstrap_stats = []
        n = len(data)

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))

        bootstrap_stats = np.array(bootstrap_stats)
        point_estimate = statistic_func(data)

        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return point_estimate, (ci_lower, ci_upper)

    def correct_multiple_comparisons(
        self,
        p_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple comparison correction to p-values.

        Args:
            p_values: Array of p-values

        Returns:
            Tuple of (rejected null hypotheses, corrected p-values)
        """
        if self.correction_method == 'none':
            rejected = p_values < self.significance_level
            return rejected, p_values

        rejected, corrected_p, _, _ = multipletests(
            p_values,
            alpha=self.significance_level,
            method=self.correction_method
        )

        logger.info(
            f"Multiple comparison correction ({self.correction_method}): "
            f"{np.sum(rejected)}/{len(p_values)} tests significant"
        )

        return rejected, corrected_p

    def analyze_diagnosis_shifts(
        self,
        baseline_probs: pd.DataFrame,
        treatment_probs: pd.DataFrame,
        diagnosis_codes: List[str],
        use_permutation: bool = True,
        n_permutations: int = 10000,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Comprehensive analysis of diagnosis probability shifts.

        Args:
            baseline_probs: Baseline diagnosis probabilities (samples × diagnoses)
            treatment_probs: Treatment diagnosis probabilities (samples × diagnoses)
            diagnosis_codes: List of diagnosis codes
            use_permutation: Whether to include permutation tests (default: True)
            n_permutations: Number of permutations for randomization tests
            random_seed: Random seed for permutation tests

        Returns:
            DataFrame with analysis results per diagnosis
        """
        results = []

        for code in diagnosis_codes:
            if code not in baseline_probs.columns or code not in treatment_probs.columns:
                logger.warning(f"Diagnosis code {code} not found in data")
                continue

            baseline = baseline_probs[code].values
            treatment = treatment_probs[code].values

            # Perform traditional parametric tests
            ttest_result = self.paired_ttest(baseline, treatment)
            wilcoxon_result = self.wilcoxon_test(baseline, treatment)

            # Perform permutation test (approximate randomization)
            if use_permutation:
                perm_result = self.paired_permutation_test(
                    baseline,
                    treatment,
                    n_permutations=n_permutations,
                    random_seed=random_seed
                )

            # Calculate additional statistics
            mean_shift = np.mean(treatment - baseline)
            median_shift = np.median(treatment - baseline)
            hedges_g = self.hedges_g(treatment, baseline, paired=True)

            # Bootstrap CI for mean shift
            _, ci_mean_shift = self.bootstrap_confidence_interval(
                treatment - baseline,
                np.mean,
                n_bootstrap=5000
            )

            result_dict = {
                'diagnosis_code': code,
                'mean_shift': mean_shift,
                'median_shift': median_shift,
                'cohens_d': ttest_result.effect_size,
                'hedges_g': hedges_g,
                'ttest_statistic': ttest_result.statistic,
                'ttest_pvalue': ttest_result.p_value,
                'wilcoxon_statistic': wilcoxon_result.statistic,
                'wilcoxon_pvalue': wilcoxon_result.p_value,
                'ci_lower': ci_mean_shift[0],
                'ci_upper': ci_mean_shift[1],
                'baseline_mean': np.mean(baseline),
                'baseline_std': np.std(baseline),
                'treatment_mean': np.mean(treatment),
                'treatment_std': np.std(treatment)
            }

            # Add permutation test results if enabled
            if use_permutation:
                result_dict.update({
                    'permutation_pvalue': perm_result.p_value,
                    'permutation_n_permutations': n_permutations
                })

            results.append(result_dict)

        results_df = pd.DataFrame(results)

        # Apply multiple comparison correction
        if len(results_df) > 1:
            rejected_ttest, corrected_p_ttest = self.correct_multiple_comparisons(
                results_df['ttest_pvalue'].values
            )
            rejected_wilcoxon, corrected_p_wilcoxon = self.correct_multiple_comparisons(
                results_df['wilcoxon_pvalue'].values
            )

            results_df['ttest_pvalue_corrected'] = corrected_p_ttest
            results_df['ttest_significant'] = rejected_ttest
            results_df['wilcoxon_pvalue_corrected'] = corrected_p_wilcoxon
            results_df['wilcoxon_significant'] = rejected_wilcoxon

            # Apply correction to permutation p-values if enabled
            if use_permutation and 'permutation_pvalue' in results_df.columns:
                rejected_perm, corrected_p_perm = self.correct_multiple_comparisons(
                    results_df['permutation_pvalue'].values
                )
                results_df['permutation_pvalue_corrected'] = corrected_p_perm
                results_df['permutation_significant'] = rejected_perm

                logger.info(
                    f"Permutation tests: {np.sum(rejected_perm)}/{len(results_df)} "
                    f"diagnoses significant after correction"
                )

        return results_df

    def analyze_attention_shifts(
        self,
        baseline_attention: pd.DataFrame,
        treatment_attention: pd.DataFrame,
        words: List[str],
        top_n: int = 50,
        use_permutation: bool = True,
        n_permutations: int = 10000,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Analyze attention weight shifts for words.

        Args:
            baseline_attention: Baseline attention weights (samples × words)
            treatment_attention: Treatment attention weights (samples × words)
            words: List of words to analyze
            top_n: Number of top words to return
            use_permutation: Whether to include permutation tests (default: True)
            n_permutations: Number of permutations for randomization tests
            random_seed: Random seed for permutation tests

        Returns:
            DataFrame with attention shift analysis
        """
        results = []

        for word in words:
            if word not in baseline_attention.columns or word not in treatment_attention.columns:
                continue

            baseline = baseline_attention[word].values
            treatment = treatment_attention[word].values

            # Calculate shift statistics
            mean_shift = np.mean(treatment - baseline)
            effect_size = self.cohens_d(treatment, baseline, paired=True)

            # Test significance with parametric test
            ttest_result = self.paired_ttest(baseline, treatment)

            result_dict = {
                'word': word,
                'mean_shift': mean_shift,
                'abs_mean_shift': abs(mean_shift),
                'cohens_d': effect_size,
                'ttest_pvalue': ttest_result.p_value,
                'baseline_mean': np.mean(baseline),
                'treatment_mean': np.mean(treatment)
            }

            # Add permutation test if enabled
            if use_permutation:
                perm_result = self.paired_permutation_test(
                    baseline,
                    treatment,
                    n_permutations=n_permutations,
                    random_seed=random_seed
                )
                result_dict['permutation_pvalue'] = perm_result.p_value

            results.append(result_dict)

        results_df = pd.DataFrame(results)

        # Sort by absolute mean shift and take top N
        results_df = results_df.sort_values('abs_mean_shift', ascending=False).head(top_n)

        # Apply multiple comparison correction
        if len(results_df) > 1:
            rejected, corrected_p = self.correct_multiple_comparisons(
                results_df['ttest_pvalue'].values
            )
            results_df['ttest_pvalue_corrected'] = corrected_p
            results_df['significant'] = rejected

            # Apply correction to permutation p-values if enabled
            if use_permutation and 'permutation_pvalue' in results_df.columns:
                rejected_perm, corrected_p_perm = self.correct_multiple_comparisons(
                    results_df['permutation_pvalue'].values
                )
                results_df['permutation_pvalue_corrected'] = corrected_p_perm
                results_df['permutation_significant'] = rejected_perm

        return results_df

    def summary_statistics(
        self,
        data: pd.DataFrame,
        group_col: str,
        value_cols: List[str]
    ) -> pd.DataFrame:
        """
        Calculate summary statistics by group.

        Args:
            data: Input DataFrame
            group_col: Column name for grouping
            value_cols: List of column names for which to calculate statistics

        Returns:
            DataFrame with summary statistics
        """
        summary_funcs = {
            'count': 'count',
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            'q25': lambda x: x.quantile(0.25),
            'median': 'median',
            'q75': lambda x: x.quantile(0.75),
            'max': 'max'
        }

        results = []
        for val_col in value_cols:
            grouped = data.groupby(group_col)[val_col].agg(**summary_funcs)
            grouped['variable'] = val_col
            results.append(grouped.reset_index())

        return pd.concat(results, ignore_index=True)

    def effect_size_interpretation(self, effect_size: float) -> str:
        """
        Interpret effect size magnitude (Cohen's d or Hedges' g).

        Args:
            effect_size: Effect size value

        Returns:
            Interpretation string
        """
        abs_es = abs(effect_size)
        if abs_es < 0.2:
            return "negligible"
        elif abs_es < 0.5:
            return "small"
        elif abs_es < 0.8:
            return "medium"
        else:
            return "large"

    def generate_analysis_report(
        self,
        diagnosis_results: pd.DataFrame,
        attention_results: Optional[pd.DataFrame] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate comprehensive analysis report.

        Args:
            diagnosis_results: Results from analyze_diagnosis_shifts
            attention_results: Results from analyze_attention_shifts (optional)
            output_path: Path to save report (optional)

        Returns:
            Report as string
        """
        # Check if permutation tests were used
        has_permutation = 'permutation_pvalue' in diagnosis_results.columns

        report_lines = [
            "=" * 80,
            "STATISTICAL ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Significance level: {self.significance_level}",
            f"Multiple comparison correction: {self.correction_method}",
        ]

        if has_permutation:
            n_perm = diagnosis_results['permutation_n_permutations'].iloc[0] if 'permutation_n_permutations' in diagnosis_results.columns else 'N/A'
            report_lines.append(f"Permutation testing: ENABLED (n_permutations={n_perm})")
            report_lines.append("Reference: Yeh (2000) - Approximate Randomization")

        report_lines.extend([
            "",
            "=" * 80,
            "DIAGNOSIS PROBABILITY SHIFTS",
            "=" * 80,
            ""
        ])

        # Significant diagnoses - Parametric tests
        if 'ttest_significant' in diagnosis_results.columns:
            sig_diagnoses = diagnosis_results[diagnosis_results['ttest_significant']]
            report_lines.append(f"Significant diagnoses - Paired t-test (after correction): {len(sig_diagnoses)}/{len(diagnosis_results)}")
            report_lines.append("")

            if len(sig_diagnoses) > 0:
                report_lines.append("Top 10 most affected diagnoses (t-test):")
                # Create abs column for sorting (nlargest doesn't support key parameter)
                sig_diagnoses_copy = sig_diagnoses.copy()
                sig_diagnoses_copy['abs_mean_shift'] = sig_diagnoses_copy['mean_shift'].abs()
                top_diagnoses = sig_diagnoses_copy.nlargest(10, 'abs_mean_shift')

                for _, row in top_diagnoses.iterrows():
                    effect_interp = self.effect_size_interpretation(row['cohens_d'])
                    report_lines.append(
                        f"  {row['diagnosis_code']}: "
                        f"mean_shift={row['mean_shift']:.6f}, "
                        f"d={row['cohens_d']:.3f} ({effect_interp}), "
                        f"p_ttest={row['ttest_pvalue_corrected']:.4e}"
                    )
                report_lines.append("")

        # Significant diagnoses - Permutation tests
        if has_permutation and 'permutation_significant' in diagnosis_results.columns:
            sig_diagnoses_perm = diagnosis_results[diagnosis_results['permutation_significant']]
            report_lines.append(f"Significant diagnoses - Permutation test (after correction): {len(sig_diagnoses_perm)}/{len(diagnosis_results)}")
            report_lines.append("")

            if len(sig_diagnoses_perm) > 0:
                report_lines.append("Top 10 most affected diagnoses (permutation test):")
                # Create abs column for sorting (nlargest doesn't support key parameter)
                sig_diagnoses_perm_copy = sig_diagnoses_perm.copy()
                sig_diagnoses_perm_copy['abs_mean_shift'] = sig_diagnoses_perm_copy['mean_shift'].abs()
                top_diagnoses_perm = sig_diagnoses_perm_copy.nlargest(10, 'abs_mean_shift')

                for _, row in top_diagnoses_perm.iterrows():
                    effect_interp = self.effect_size_interpretation(row['cohens_d'])
                    report_lines.append(
                        f"  {row['diagnosis_code']}: "
                        f"mean_shift={row['mean_shift']:.6f}, "
                        f"d={row['cohens_d']:.3f} ({effect_interp}), "
                        f"p_perm={row['permutation_pvalue_corrected']:.4e}"
                    )
                report_lines.append("")

            # Comparison of t-test vs permutation test
            if 'ttest_significant' in diagnosis_results.columns:
                ttest_sig = set(diagnosis_results[diagnosis_results['ttest_significant']]['diagnosis_code'])
                perm_sig = set(diagnosis_results[diagnosis_results['permutation_significant']]['diagnosis_code'])

                agreement = len(ttest_sig & perm_sig)
                ttest_only = len(ttest_sig - perm_sig)
                perm_only = len(perm_sig - ttest_sig)

                report_lines.extend([
                    "Comparison: t-test vs Permutation test",
                    f"  Agreement (both significant): {agreement}",
                    f"  t-test only: {ttest_only}",
                    f"  Permutation only: {perm_only}",
                    ""
                ])

        # Attention shifts
        if attention_results is not None:
            has_perm_attention = 'permutation_pvalue' in attention_results.columns

            report_lines.extend([
                "=" * 80,
                "ATTENTION WEIGHT SHIFTS",
                "=" * 80,
                ""
            ])

            # Parametric test results
            if 'significant' in attention_results.columns:
                sig_words = attention_results[attention_results['significant']]
                report_lines.append(f"Significant words - t-test: {len(sig_words)}/{len(attention_results)}")
                report_lines.append("")

                if len(sig_words) > 0:
                    report_lines.append("Top 10 words with largest attention shifts (t-test):")
                    for _, row in sig_words.head(10).iterrows():
                        effect_interp = self.effect_size_interpretation(row['cohens_d'])
                        line = (
                            f"  '{row['word']}': "
                            f"shift={row['mean_shift']:.6f}, "
                            f"d={row['cohens_d']:.3f} ({effect_interp}), "
                            f"p_ttest={row['ttest_pvalue_corrected']:.4e}"
                        )
                        if has_perm_attention:
                            line += f", p_perm={row.get('permutation_pvalue_corrected', 'N/A'):.4e}"
                        report_lines.append(line)
                    report_lines.append("")

            # Permutation test results
            if has_perm_attention and 'permutation_significant' in attention_results.columns:
                sig_words_perm = attention_results[attention_results['permutation_significant']]
                report_lines.append(f"Significant words - Permutation test: {len(sig_words_perm)}/{len(attention_results)}")
                report_lines.append("")

                # Comparison
                if 'significant' in attention_results.columns:
                    ttest_sig_words = set(attention_results[attention_results['significant']]['word'])
                    perm_sig_words = set(attention_results[attention_results['permutation_significant']]['word'])

                    agreement = len(ttest_sig_words & perm_sig_words)
                    ttest_only = len(ttest_sig_words - perm_sig_words)
                    perm_only = len(perm_sig_words - ttest_sig_words)

                    report_lines.extend([
                        "Comparison: t-test vs Permutation test",
                        f"  Agreement (both significant): {agreement}",
                        f"  t-test only: {ttest_only}",
                        f"  Permutation only: {perm_only}",
                        ""
                    ])

        report_lines.append("")
        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Analysis report saved to {output_path}")

        return report
