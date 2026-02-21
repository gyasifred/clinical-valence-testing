#!/usr/bin/env python3
"""
analyze_valence_results.py
==========================
Standalone statistical analysis script for Clinical Valence Behavioral Testing.

Reads prediction CSVs produced by main.py, runs all statistical comparisons
between the neutralized baseline and each valence shift condition, generates
publication-quality figures, and writes a structured summary report.

Usage
-----
python analyze_valence_results.py \
    --results_dir ./results \
    --baseline_key neutralize \
    --n_permutations 10000 \
    --alpha 0.05 \
    --correction fdr_bh \
    --output_dir ./results/analysis

Research Questions Addressed
-----------------------------
RQ1: Pejorative shift — do negative descriptors alter ICD-9 predictions?
RQ2: Laudatory shift — do positive descriptors alter ICD-9 predictions?
RQ3: Neutral valence shift — do neutral descriptors alter ICD-9 predictions?
RQ4: Cross-condition asymmetry — do pejorative and laudatory shift in opposite directions?
RQ5: Attention — do attention weights on valence terms shift across conditions?

Statistical Methods (Yeh 2000; Heider 2023)
--------------------------------------------
- Primary: Paired permutation test (approximate randomization, T=10,000)
- Supplementary: Paired t-test, Wilcoxon signed-rank test
- Effect size: Cohen's d, Hedges' g
- CI: Bootstrap 95% CI (B=5,000)
- Correction: Benjamini-Hochberg FDR (Benjamini & Hochberg 1995)
"""

import os
import re
import sys
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NON_CODE_COLS = {
    "NoteID", "Valence", "Val_class", "text", "shifted_text",
    "sample_id", "group", "attention_weights",
}

# ICD-9 chapter boundaries for annotation
ICD9_CHAPTERS = {
    (1,   139): "Infectious & Parasitic",
    (140, 239): "Neoplasms",
    (240, 279): "Endocrine / Metabolic",
    (280, 289): "Blood",
    (290, 319): "Mental Disorders",
    (320, 389): "Nervous System",
    (390, 459): "Circulatory",
    (460, 519): "Respiratory",
    (520, 579): "Digestive",
    (580, 629): "Genitourinary",
    (630, 679): "Pregnancy",
    (680, 709): "Skin",
    (710, 739): "Musculoskeletal",
    (740, 759): "Congenital",
    (760, 779): "Perinatal",
    (780, 799): "Symptoms / Signs",
    (800, 999): "Injury & Poisoning",
}

CONDITION_PALETTE = {
    "neutralize": "#4e79a7",
    "pejorative": "#e15759",
    "laud":       "#59a14f",
    "laudatory":  "#59a14f",
    "neutralval": "#f28e2b",
}

EFFECT_THRESHOLDS = {
    # Lower-bound values for each Cohen's d category (Cohen 1988).
    # |d| < 0.2           → negligible
    # 0.2 ≤ |d| < 0.5    → small
    # 0.5 ≤ |d| < 0.8    → medium
    # |d| ≥ 0.8           → large
    "small":  0.2,
    "medium": 0.5,
    "large":  0.8,
}


# ===========================================================================
# Utility helpers
# ===========================================================================

def icd9_chapter(code: str) -> str:
    """Map a 3-digit ICD-9 code string to its chapter label."""
    if code.startswith(("V", "v")):
        return "V Codes"
    if code.startswith(("E", "e")):
        return "E Codes"
    try:
        n = int(code)
        for (lo, hi), label in ICD9_CHAPTERS.items():
            if lo <= n <= hi:
                return label
    except ValueError:
        pass
    return "Other"


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude label (Cohen 1988 thresholds)."""
    a = abs(d)
    if a < EFFECT_THRESHOLDS["small"]:
        return "negligible"
    if a < EFFECT_THRESHOLDS["medium"]:
        return "small"
    if a < EFFECT_THRESHOLDS["large"]:
        return "medium"
    return "large"


def find_csv_for_condition(results_dir: Path, condition: str) -> Optional[Path]:
    """
    Locate the CSV file for a given shift condition.
    Handles both timestamped filenames (e.g. neutralize_20250101_123456_diagnosis.csv)
    and simple filenames (e.g. neutralize_shift_diagnosis.csv).
    """
    shift_prefix_map = {
        "neutralize": "neutralize",
        "pejorative": "pejorative",
        "laud":       "laudatory",
        "laudatory":  "laudatory",
        "neutralval": "neutralval",
    }
    prefix = shift_prefix_map.get(condition, condition)

    patterns = [
        rf"^{re.escape(prefix)}_\d{{8}}_\d{{6}}_diagnosis\.csv$",
        rf"^{re.escape(condition)}_shift_diagnosis\.csv$",
        rf"^{re.escape(prefix)}_.*diagnosis\.csv$",
        rf"^{re.escape(prefix)}_diagnosis\.csv$",
    ]

    for filename in os.listdir(results_dir):
        for pat in patterns:
            if re.match(pat, filename, re.IGNORECASE):
                return results_dir / filename
    return None


def load_predictions(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load a prediction CSV and return (dataframe, list_of_code_columns).
    Code columns are all numeric columns that are not metadata.
    """
    df = pd.read_csv(path)
    code_cols = [
        c for c in df.columns
        if c not in NON_CODE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    logger.info(f"  Loaded {path.name}: {len(df)} notes, {len(code_cols)} ICD-9 codes")
    return df, code_cols


# ===========================================================================
# Core statistical functions
# ===========================================================================

def cohens_d_paired(baseline: np.ndarray, treatment: np.ndarray) -> float:
    """Cohen's d for paired samples (d = mean(diff) / SD(diff))."""
    diff = treatment - baseline
    sd = np.std(diff, ddof=1)
    return 0.0 if sd == 0 or np.isnan(sd) else float(np.mean(diff) / sd)


def hedges_g_paired(baseline: np.ndarray, treatment: np.ndarray) -> float:
    """
    Bias-corrected Hedges' g for paired samples.

    For paired designs, df = n - 1, so the correction factor is:
        J(df) = 1 - 3 / (4 * df - 1)
              = 1 - 3 / (4 * (n-1) - 1)
              = 1 - 3 / (4n - 5)
    This is the correct paired-sample formula (Hedges & Olkin 1985).
    The commonly seen 4n-9 denominator applies to two-independent-sample designs
    where N is the *total* sample size (df = N - 2), which is incorrect here.
    """
    d = cohens_d_paired(baseline, treatment)
    n = len(baseline)
    if n < 3:                        # correction undefined for n < 3 (df < 2)
        return d
    return d * (1.0 - 3.0 / (4.0 * n - 5.0))


def bootstrap_ci(
    diffs: np.ndarray,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap 95% CI for the mean of diffs.
    Returns (lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def paired_permutation_test(
    baseline: np.ndarray,
    treatment: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """
    Approximate randomization test (Yeh 2000).
    H0: mean(treatment - baseline) = 0
    Returns two-sided p-value with Laplace smoothing.
    """
    rng = np.random.default_rng(seed)
    diffs = treatment - baseline
    obs = np.abs(np.mean(diffs))

    # Vectorized: draw (n_perm x n) sign-flip matrix
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(diffs)))
    perm_means = np.abs((signs * diffs[np.newaxis, :]).mean(axis=1))
    count_extreme = int(np.sum(perm_means >= obs))

    return (count_extreme + 1) / (n_permutations + 1)


def analyze_one_code(
    code: str,
    baseline: np.ndarray,
    treatment: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict:
    """Run all tests for a single ICD-9 code. Returns a result dict."""
    diffs = treatment - baseline

    # Paired t-test
    t_stat, t_pval = ttest_rel(baseline, treatment)

    # Wilcoxon signed-rank (skip if all diffs are zero)
    if np.all(diffs == 0):
        w_stat, w_pval = np.nan, 1.0
    else:
        try:
            w_stat, w_pval = wilcoxon(diffs, zero_method="wilcox")
        except Exception:
            w_stat, w_pval = np.nan, 1.0

    # Permutation test
    perm_pval = paired_permutation_test(baseline, treatment,
                                        n_permutations=n_permutations,
                                        seed=seed)

    # Effect sizes
    d = cohens_d_paired(baseline, treatment)
    g = hedges_g_paired(baseline, treatment)

    # Bootstrap CI
    ci_lo, ci_hi = bootstrap_ci(diffs, seed=seed)

    return {
        "diagnosis_code":       code,
        "icd9_chapter":         icd9_chapter(code),
        "n_samples":            len(baseline),
        "mean_shift":           float(np.mean(diffs)),
        "median_shift":         float(np.median(diffs)),
        "std_shift":            float(np.std(diffs, ddof=1)),
        "ci_lower":             ci_lo,
        "ci_upper":             ci_hi,
        "baseline_mean":        float(np.mean(baseline)),
        "baseline_std":         float(np.std(baseline, ddof=1)),
        "treatment_mean":       float(np.mean(treatment)),
        "treatment_std":        float(np.std(treatment, ddof=1)),
        "cohens_d":             d,
        "hedges_g":             g,
        "effect_size_label":    interpret_effect_size(d),
        "ttest_statistic":      float(t_stat),
        "ttest_pvalue":         float(t_pval),
        "wilcoxon_statistic":   float(w_stat) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_pvalue":      float(w_pval),
        "permutation_pvalue":   float(perm_pval),
        "permutation_n":        n_permutations,
    }


def apply_fdr_correction(
    results_df: pd.DataFrame,
    pval_col: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Add corrected p-values and significance flag columns for a given raw p-value column.
    Uses Benjamini-Hochberg FDR (Benjamini & Hochberg 1995).
    """
    pvals = results_df[pval_col].fillna(1.0).values
    rejected, corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    out_col  = pval_col + "_corrected"
    flag_col = pval_col.replace("_pvalue", "") + "_significant"
    results_df = results_df.copy()
    results_df[out_col]  = corrected
    results_df[flag_col] = rejected
    return results_df


def run_full_comparison(
    baseline_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    code_cols: List[str],
    n_permutations: int,
    alpha: float,
    seed: int,
    label: str,
) -> pd.DataFrame:
    """
    Run all statistical tests for every code column.
    Returns a fully annotated results DataFrame.
    """
    logger.info(f"  Running {label}: {len(code_cols)} codes × {len(baseline_df)} notes ...")

    # --- Align notes by NoteID to guarantee correct pairing ----------------
    # .head(n) would silently mis-pair notes if CSVs are in different orders.
    # We must match on the note identifier column explicitly.
    note_id_col = next(
        (c for c in ("NoteID", "sample_id", "note_id") if c in baseline_df.columns),
        None,
    )
    if note_id_col is not None and note_id_col in treatment_df.columns:
        base_idx  = baseline_df.set_index(note_id_col)
        treat_idx = treatment_df.set_index(note_id_col)
        common_ids = base_idx.index.intersection(treat_idx.index)
        if len(common_ids) == 0:
            logger.error("  No overlapping NoteIDs between baseline and treatment!")
            return pd.DataFrame()
        base_aligned  = base_idx.loc[common_ids]
        treat_aligned = treat_idx.loc[common_ids]
        n_aligned = len(common_ids)
        if n_aligned < len(baseline_df):
            logger.warning(f"  NoteID alignment: {len(baseline_df)} base rows → {n_aligned} matched pairs")
        logger.info(f"  Aligned on NoteID: {n_aligned} paired notes")
    else:
        # Fallback: assume same-order CSVs and truncate to min length
        logger.warning(
            "  NoteID column not found — assuming CSVs are in the same row order. "
            "Verify this holds for your data."
        )
        n_aligned = min(len(baseline_df), len(treatment_df))
        base_aligned  = baseline_df.head(n_aligned)
        treat_aligned = treatment_df.head(n_aligned)

    shared_codes = [c for c in code_cols if c in treat_aligned.columns]
    logger.info(f"  Shared ICD-9 codes: {len(shared_codes)}")
    rows = []
    for i, code in enumerate(shared_codes):
        if i % 200 == 0:
            logger.info(f"    {i}/{len(shared_codes)} codes processed")
        b = base_aligned[code].fillna(0).values
        t = treat_aligned[code].fillna(0).values
        # Unique seed per code: prevents all 1266 tests from drawing
        # identical sign-flip matrices, which would create correlated null
        # distributions and undermine FDR correction validity.
        code_seed = seed + i
        rows.append(analyze_one_code(code, b, t,
                                     n_permutations=n_permutations,
                                     seed=code_seed))

    results = pd.DataFrame(rows)

    # Apply FDR correction to all three test p-values
    for pcol in ["ttest_pvalue", "wilcoxon_pvalue", "permutation_pvalue"]:
        results = apply_fdr_correction(results, pcol, alpha=alpha)

    # Sort by absolute mean shift
    results = results.sort_values("mean_shift", key=lambda x: x.abs(),
                                  ascending=False).reset_index(drop=True)
    return results


# ===========================================================================
# Cross-condition analysis (RQ4)
# ===========================================================================

def cross_condition_summary(
    all_results: Dict[str, pd.DataFrame],
    alpha: float,
) -> pd.DataFrame:
    """
    Merge all condition results on diagnosis_code.
    Returns a wide DataFrame with mean_shift per condition.
    """
    frames = []
    for cond, df in all_results.items():
        sig_col = "permutation_significant" if "permutation_significant" in df.columns else None
        keep = ["diagnosis_code", "mean_shift", "cohens_d"]
        if sig_col:
            keep.append(sig_col)

        sub = df[keep].copy()
        new_names = {
            "mean_shift": f"mean_shift_{cond}",
            "cohens_d":   f"cohens_d_{cond}",
        }
        if sig_col:
            new_names[sig_col] = f"sig_{cond}"
        sub = sub.rename(columns=new_names)
        frames.append(sub.set_index("diagnosis_code"))

    combined = pd.concat(frames, axis=1)
    combined.index.name = "diagnosis_code"
    return combined.reset_index()


def asymmetry_stats(
    pej_df: pd.DataFrame,
    laud_df: pd.DataFrame,
) -> dict:
    """
    Compute summary statistics for pejorative vs. laudatory asymmetry (RQ4).
    """
    merged = pej_df[["diagnosis_code", "mean_shift"]].merge(
        laud_df[["diagnosis_code", "mean_shift"]],
        on="diagnosis_code", suffixes=("_pej", "_laud")
    ).dropna()

    r, p = stats.pearsonr(merged["mean_shift_pej"], merged["mean_shift_laud"])
    same_dir = ((merged["mean_shift_pej"] * merged["mean_shift_laud"]) > 0).sum()
    opp_dir  = ((merged["mean_shift_pej"] * merged["mean_shift_laud"]) < 0).sum()

    return {
        "n_codes":              len(merged),
        "pearson_r":            round(r, 4),
        "pearson_p":            round(p, 6),
        "same_direction_codes": int(same_dir),
        "opposite_direction_codes": int(opp_dir),
        "pct_opposite":         round(100 * opp_dir / len(merged), 1),
    }


# ===========================================================================
# Visualisation
# ===========================================================================

def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {path.name}")


def plot_shift_distribution(
    results: pd.DataFrame,
    condition: str,
    output_dir: Path,
) -> None:
    """Histogram of mean probability shifts for all ICD-9 codes."""
    color = CONDITION_PALETTE.get(condition, "steelblue")
    sig_col = "permutation_significant"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All codes
    axes[0].hist(results["mean_shift"], bins=80,
                 color=color, edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="black", lw=1.5)
    axes[0].axvline(0.01,  color="crimson",   lw=1, ls="--", label="+0.01")
    axes[0].axvline(-0.01, color="navy", lw=1, ls="--", label="−0.01")
    axes[0].set_xlabel("Mean Probability Shift (condition − baseline)")
    axes[0].set_ylabel("Number of ICD-9 Codes")
    axes[0].set_title(f"{condition.capitalize()} — All Codes")
    axes[0].legend(fontsize=9)

    # Significant only
    if sig_col in results.columns:
        sig_vals = results.loc[results[sig_col] == True, "mean_shift"]
        axes[1].hist(sig_vals, bins=40,
                     color=color, edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="black", lw=1.5)
        axes[1].set_xlabel("Mean Probability Shift (Significant Codes)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Significant Codes (n={len(sig_vals):,})")

    fig.suptitle(
        f"Probability Shift Distribution — {condition.capitalize()} vs. Neutralized",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    _save(fig, output_dir / f"fig_{condition}_shift_dist.png")


def plot_volcano(
    results: pd.DataFrame,
    condition: str,
    output_dir: Path,
) -> None:
    """Volcano plot: mean shift vs. −log10(corrected permutation p-value)."""
    if "permutation_pvalue_corrected" not in results.columns:
        return
    color = CONDITION_PALETTE.get(condition, "steelblue")

    df = results.copy()
    df["neg_log_p"] = -np.log10(
        df["permutation_pvalue_corrected"].clip(lower=1e-6)
    )
    df["sig"] = df.get("permutation_significant", False).fillna(False)
    threshold = -np.log10(0.05)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = df["sig"].map({True: color, False: "#d0d0d0"})
    ax.scatter(df["mean_shift"], df["neg_log_p"],
               c=colors, alpha=0.55, s=15, linewidths=0)
    ax.axhline(threshold, color="black", ls="--", lw=1.0, label="FDR α = 0.05")
    ax.axvline(0, color="gray", lw=0.7)

    # Annotate top 12 by −log p
    top = df[df["sig"]].nlargest(12, "neg_log_p")
    for _, row in top.iterrows():
        ax.annotate(
            row["diagnosis_code"],
            xy=(row["mean_shift"], row["neg_log_p"]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=7, color="black",
        )

    ax.set_xlabel("Mean Probability Shift (condition − baseline)", fontsize=11)
    ax.set_ylabel("−log₁₀(Corrected p-value)", fontsize=11)
    ax.set_title(
        f"Volcano Plot — {condition.capitalize()} vs. Neutralized", fontsize=13
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir / f"fig_{condition}_volcano.png")


def plot_top_codes_bar(
    results: pd.DataFrame,
    condition: str,
    output_dir: Path,
    top_n: int = 25,
) -> None:
    """Horizontal bar chart of top N most shifted (significant) codes."""
    color = CONDITION_PALETTE.get(condition, "steelblue")
    sig_col = "permutation_significant"

    if sig_col in results.columns:
        sig = results[results[sig_col] == True].copy()
    else:
        sig = results.copy()

    if sig.empty:
        logger.warning(f"  No significant codes for {condition} — skipping bar plot")
        return

    top = sig.nlargest(top_n, "mean_shift", keep="all").head(top_n)
    bottom = sig.nsmallest(top_n, "mean_shift", keep="all").head(top_n)
    combined = pd.concat([top, bottom]).drop_duplicates("diagnosis_code")
    combined = combined.sort_values("mean_shift")

    bar_colors = [color if v > 0 else "#aaaaaa" for v in combined["mean_shift"]]

    fig, ax = plt.subplots(figsize=(10, max(8, len(combined) * 0.32)))
    ax.barh(combined["diagnosis_code"], combined["mean_shift"],
            color=bar_colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Mean Probability Shift")
    ax.set_title(
        f"Top Shifted ICD-9 Codes — {condition.capitalize()} vs. Neutralized\n"
        f"(FDR-corrected permutation test, α=0.05)",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, output_dir / f"fig_{condition}_top_codes.png")


def plot_cross_condition_heatmap(
    combined: pd.DataFrame,
    shift_cols: List[str],
    output_dir: Path,
    top_n: int = 40,
) -> None:
    """Heatmap of mean shifts across all conditions for the most affected codes."""
    heat = combined.set_index("diagnosis_code")[shift_cols].copy()
    heat["max_abs"] = heat.abs().max(axis=1)
    top = heat.nlargest(top_n, "max_abs").drop(columns="max_abs")

    # Rename columns for display
    display_cols = [c.replace("mean_shift_", "").capitalize() for c in shift_cols]
    top.columns = display_cols

    fig, ax = plt.subplots(figsize=(len(display_cols) * 2.5 + 3, max(12, top_n * 0.32)))
    sns.heatmap(
        top,
        cmap="vlag", center=0,
        vmin=-0.035, vmax=0.035,
        linewidths=0.3,
        annot=len(top) <= 30,
        fmt=".4f" if len(top) <= 30 else "",
        ax=ax,
        cbar_kws={"label": "Mean Probability Shift (condition − baseline)"},
    )
    ax.set_title(
        f"Cross-Condition Probability Shifts\n(Top {top_n} ICD-9 Codes by Max |Shift|)",
        fontsize=13,
    )
    ax.set_xlabel("Valence Condition")
    ax.set_ylabel("ICD-9 Code")
    plt.tight_layout()
    _save(fig, output_dir / "fig_cross_condition_heatmap.png")


def plot_asymmetry_scatter(
    pej_df: pd.DataFrame,
    laud_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Scatter of pejorative vs. laudatory shift per ICD-9 code (RQ4)."""
    merged = pej_df[["diagnosis_code", "mean_shift"]].merge(
        laud_df[["diagnosis_code", "mean_shift"]],
        on="diagnosis_code", suffixes=("_pej", "_laud")
    ).dropna()

    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(merged["mean_shift_laud"], merged["mean_shift_pej"],
               alpha=0.30, s=12, color="#555555")

    lim = max(merged[["mean_shift_pej", "mean_shift_laud"]].abs().max().max() * 1.15, 0.02)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0, label="y = x (symmetric)")
    ax.plot([-lim, lim], [ lim, -lim], color="gray", lw=0.7, ls=":")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Laudatory Mean Shift", fontsize=11)
    ax.set_ylabel("Pejorative Mean Shift", fontsize=11)

    r, p = stats.pearsonr(merged["mean_shift_pej"], merged["mean_shift_laud"])
    ax.set_title(
        f"Pejorative vs. Laudatory Shift Asymmetry\n"
        f"(n={len(merged):,} codes, r={r:.3f}, p={p:.4f})",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir / "fig_rq4_asymmetry_scatter.png")


def plot_chapter_summary(
    results: pd.DataFrame,
    condition: str,
    output_dir: Path,
) -> None:
    """Box plot of mean shifts grouped by ICD-9 chapter."""
    sig_col = "permutation_significant"
    color = CONDITION_PALETTE.get(condition, "steelblue")

    if "icd9_chapter" not in results.columns:
        return

    chapter_order = (
        results.groupby("icd9_chapter")["mean_shift"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    data_by_chapter = [
        results.loc[results["icd9_chapter"] == ch, "mean_shift"].values
        for ch in chapter_order
    ]

    bp = ax.boxplot(data_by_chapter, patch_artist=True, vert=True,
                    medianprops={"color": "black", "lw": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.set_xticks(range(1, len(chapter_order) + 1))
    ax.set_xticklabels(chapter_order, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Mean Probability Shift")
    ax.set_title(
        f"Shift Distribution by ICD-9 Chapter — {condition.capitalize()} vs. Neutralized",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, output_dir / f"fig_{condition}_chapter_boxplot.png")


def plot_effect_size_distribution(
    results: pd.DataFrame,
    condition: str,
    output_dir: Path,
) -> None:
    """Histogram of Cohen's d values for all significant codes."""
    color = CONDITION_PALETTE.get(condition, "steelblue")
    sig_col = "permutation_significant"

    df = results[results.get(sig_col, pd.Series(False)) == True] if sig_col in results else results

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["cohens_d"], bins=50, color=color, edgecolor="white", alpha=0.85)
    for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline( thresh, color="black", ls=":", lw=1.0, label=f"|d|={thresh} ({lbl})")
        ax.axvline(-thresh, color="black", ls=":", lw=1.0)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Cohen's d (paired)")
    ax.set_ylabel("Number of Significant ICD-9 Codes")
    ax.set_title(f"Effect Size Distribution — {condition.capitalize()} Significant Codes")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, output_dir / f"fig_{condition}_effect_sizes.png")


# ===========================================================================
# Attention analysis (RQ5)
# ===========================================================================

def load_attention_data(results_dir: Path, condition: str) -> Optional[pd.DataFrame]:
    """Try to load the attention CSV for a given condition."""
    prefix_map = {
        "pejorative": "pejorative",
        "laud":       "laudatory",
        "laudatory":  "laudatory",
        "neutralval": "neutralval",
    }
    prefix = prefix_map.get(condition, condition)
    patterns = [
        rf"^{re.escape(prefix)}_.*attention\.csv$",
        rf"^{re.escape(condition)}_.*attention\.csv$",
    ]
    for fname in os.listdir(results_dir):
        for pat in patterns:
            if re.match(pat, fname, re.IGNORECASE):
                return pd.read_csv(results_dir / fname)
    return None


def analyze_attention(
    baseline_attn: pd.DataFrame,
    treatment_attn: pd.DataFrame,
    condition: str,
    output_dir: Path,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    top_n: int = 30,
) -> Optional[pd.DataFrame]:
    """
    Compare CLS-token attention weights per word between baseline and treatment.
    Returns a results DataFrame.
    """
    # Pivot to wide format: rows=NoteID, columns=Word
    def pivot_attn(df: pd.DataFrame) -> pd.DataFrame:
        return (df.groupby(["NoteID", "Word"])["AttentionWeight"]
                  .mean()
                  .unstack(fill_value=0))

    b_wide = pivot_attn(baseline_attn)
    t_wide = pivot_attn(treatment_attn)
    common_words = b_wide.columns.intersection(t_wide.columns)
    common_notes = b_wide.index.intersection(t_wide.index)

    if len(common_words) == 0 or len(common_notes) == 0:
        logger.warning("  No overlapping words/notes for attention analysis")
        return None

    b_wide = b_wide.loc[common_notes, common_words]
    t_wide = t_wide.loc[common_notes, common_words]

    rows = []
    for word in common_words:
        b = b_wide[word].values
        t = t_wide[word].values
        diffs = t - b
        mean_shift = float(np.mean(diffs))
        d = cohens_d_paired(b, t)
        perm_p = paired_permutation_test(b, t, n_permutations=n_permutations)
        rows.append({
            "word":          word,
            "mean_shift":    mean_shift,
            "abs_shift":     abs(mean_shift),
            "cohens_d":      d,
            "perm_pvalue":   perm_p,
            "baseline_mean": float(np.mean(b)),
            "treatment_mean": float(np.mean(t)),
        })

    df = pd.DataFrame(rows).sort_values("abs_shift", ascending=False)

    # FDR correct
    rejected, corrected, _, _ = multipletests(
        df["perm_pvalue"].values, alpha=alpha, method="fdr_bh"
    )
    df["perm_pvalue_corrected"] = corrected
    df["perm_significant"] = rejected

    # Bar chart of top N
    top = df.head(top_n)
    color = CONDITION_PALETTE.get(condition, "steelblue")
    bar_colors = [color if v > 0 else "#aaaaaa" for v in top["mean_shift"]]

    fig, ax = plt.subplots(figsize=(10, max(7, top_n * 0.32)))
    ax.barh(top["word"][::-1], top["mean_shift"][::-1],
            color=bar_colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Mean Attention Weight Shift")
    ax.set_title(
        f"Top {top_n} Words — Attention Shift\n"
        f"{condition.capitalize()} vs. Neutralized (CLS, layer 11, head 11)",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, output_dir / f"fig_{condition}_attention_shift.png")

    return df


# ===========================================================================
# Report generation
# ===========================================================================

def build_report(
    all_results:      Dict[str, pd.DataFrame],
    cross_combined:   pd.DataFrame,
    asym_stats:       Optional[dict],
    alpha:            float,
    n_permutations:   int,
    output_dir:       Path,
) -> str:
    """Build a plain-text statistical report and write it to disk."""
    lines = [
        "=" * 80,
        "CLINICAL VALENCE BEHAVIORAL TESTING — STATISTICAL ANALYSIS REPORT",
        "=" * 80,
        f"Generated:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Significance level: α = {alpha}",
        f"MCC method:         Benjamini-Hochberg FDR",
        f"Permutations (T):   {n_permutations:,}",
        f"Effect size:        Cohen's d (paired) + Hedges' g",
        f"CI method:          Bootstrap percentile (B=5,000)",
        "",
    ]

    condition_labels = {
        "pejorative": "RQ1 — Pejorative Shift",
        "laud":       "RQ2 — Laudatory Shift",
        "laudatory":  "RQ2 — Laudatory Shift",
        "neutralval": "RQ3 — Neutral Valence Shift",
    }

    for cond, df in all_results.items():
        label = condition_labels.get(cond, cond)
        total = len(df)
        sig_col = "permutation_significant"
        n_sig = int(df[sig_col].sum()) if sig_col in df.columns else 0
        up    = int(((df.get(sig_col, False) == True) & (df["mean_shift"] > 0)).sum())
        down  = int(((df.get(sig_col, False) == True) & (df["mean_shift"] < 0)).sum())

        lines += [
            "=" * 80,
            label,
            "=" * 80,
            f"  Total ICD-9 codes analyzed:    {total:,}",
            f"  Significant after FDR (perm):  {n_sig:,}  ({100*n_sig/max(total,1):.1f}%)",
            f"  Upward shifts:                 {up:,}",
            f"  Downward shifts:               {down:,}",
        ]

        if n_sig > 0:
            sig = df[df.get(sig_col, pd.Series(False)) == True]
            med_d   = sig["cohens_d"].abs().median()
            max_up  = sig["mean_shift"].max()
            max_dn  = sig["mean_shift"].min()
            top_code = sig.iloc[0]["diagnosis_code"]

            lines += [
                f"  Max upward Δ:                  {max_up:+.6f}",
                f"  Max downward Δ:                {max_dn:+.6f}",
                f"  Median |Cohen's d| (sig):      {med_d:.4f}  [{interpret_effect_size(med_d)}]",
                f"  Most shifted code:             {top_code}",
                "",
                "  Top 10 significant codes by |mean_shift|:",
            ]
            top10 = sig.nlargest(10, "mean_shift", keep="all")
            for _, row in top10.iterrows():
                ci_str = f"[{row['ci_lower']:+.5f}, {row['ci_upper']:+.5f}]"
                lines.append(
                    f"    {row['diagnosis_code']:>8}  "
                    f"Δ={row['mean_shift']:+.6f}  "
                    f"95% CI {ci_str}  "
                    f"d={row['cohens_d']:+.4f}  "
                    f"p_perm={row['permutation_pvalue_corrected']:.4e}"
                )
        lines.append("")

    # RQ4 — Asymmetry
    if asym_stats:
        lines += [
            "=" * 80,
            "RQ4 — Pejorative vs. Laudatory Asymmetry",
            "=" * 80,
            f"  Codes compared:           {asym_stats['n_codes']:,}",
            f"  Pearson r:                {asym_stats['pearson_r']:.4f}  (p={asym_stats['pearson_p']:.6f})",
            f"  Same-direction shifts:    {asym_stats['same_direction_codes']:,}",
            f"  Opposite-direction shifts:{asym_stats['opposite_direction_codes']:,}  ({asym_stats['pct_opposite']:.1f}%)",
            "",
            "  Interpretation:",
            "  r ≈ −1 → mirror-image asymmetry (pejorative and laudatory fully opposed)",
            "  r ≈  0 → independent effects",
            "  r ≈ +1 → both valences shift predictions in the same direction",
            "",
        ]

    lines += [
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ]

    report = "\n".join(lines)
    report_path = output_dir / "statistical_analysis_report.txt"
    report_path.write_text(report)
    logger.info(f"Report written → {report_path.name}")
    return report


def build_excel_summary(
    all_results: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write all results to a multi-sheet Excel workbook."""
    try:
        excel_path = output_dir / "valence_analysis_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for cond, df in all_results.items():
                sheet = cond[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
                logger.info(f"  Excel sheet: {sheet}")

            # Cross-condition sheet
            # Already written in main
        logger.info(f"Excel workbook → {excel_path.name}")
    except ImportError:
        logger.warning("openpyxl not installed — skipping Excel export")


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Statistical analysis for clinical valence behavioral testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir",    type=str, default="./results",
                   help="Directory containing prediction CSVs from main.py")
    p.add_argument("--baseline_key",   type=str, default="neutralize",
                   help="Shift condition to use as baseline")
    p.add_argument("--conditions",     type=str, default="pejorative,laud,neutralval",
                   help="Comma-separated shift conditions to compare against baseline")
    p.add_argument("--n_permutations", type=int, default=10000,
                   help="Number of permutations for approximate randomization test")
    p.add_argument("--alpha",          type=float, default=0.05,
                   help="Significance level (FDR-corrected)")
    p.add_argument("--correction",     type=str, default="fdr_bh",
                   choices=["fdr_bh", "bonferroni", "none"],
                   help="Multiple comparison correction method")
    p.add_argument("--output_dir",     type=str, default=None,
                   help="Output directory (default: results_dir/analysis_<timestamp>)")
    p.add_argument("--seed",           type=int, default=42,
                   help="Random seed for permutation tests and bootstrap CI")
    p.add_argument("--top_n",          type=int, default=25,
                   help="Number of top codes to show in bar plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (
        results_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    sns.set_theme(style="whitegrid", font_scale=1.15)

    # -----------------------------------------------------------------------
    # 1.  Load baseline
    # -----------------------------------------------------------------------
    logger.info(f"Loading baseline: {args.baseline_key}")
    baseline_path = find_csv_for_condition(results_dir, args.baseline_key)
    if baseline_path is None:
        logger.error(f"Baseline CSV not found for '{args.baseline_key}' in {results_dir}")
        sys.exit(1)

    baseline_df, code_cols = load_predictions(baseline_path)

    # -----------------------------------------------------------------------
    # 2.  Run comparisons for each condition
    # -----------------------------------------------------------------------
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    all_results: Dict[str, pd.DataFrame] = {}

    for cond in conditions:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Condition: {cond}")
        path = find_csv_for_condition(results_dir, cond)
        if path is None:
            logger.warning(f"  CSV not found for '{cond}' — skipping")
            continue

        treatment_df, _ = load_predictions(path)

        results = run_full_comparison(
            baseline_df,
            treatment_df,
            code_cols,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            seed=args.seed,
            label=f"{args.baseline_key} vs {cond}",
        )

        # Save per-condition CSV
        out_csv = output_dir / f"statistical_analysis_{args.baseline_key}_vs_{cond}.csv"
        results.to_csv(out_csv, index=False)
        logger.info(f"  Results CSV → {out_csv.name}")

        all_results[cond] = results

        # ── Figures ──
        logger.info("  Generating figures ...")
        plot_shift_distribution(results, cond, output_dir)
        plot_volcano(results, cond, output_dir)
        plot_top_codes_bar(results, cond, output_dir, top_n=args.top_n)
        plot_chapter_summary(results, cond, output_dir)
        plot_effect_size_distribution(results, cond, output_dir)

        # ── Attention (RQ5) ──
        baseline_attn = load_attention_data(results_dir, args.baseline_key)
        treatment_attn = load_attention_data(results_dir, cond)
        if baseline_attn is not None and treatment_attn is not None:
            logger.info("  Running attention analysis (RQ5) ...")
            attn_results = analyze_attention(
                baseline_attn, treatment_attn, cond, output_dir,
                n_permutations=args.n_permutations, alpha=args.alpha,
            )
            if attn_results is not None:
                attn_results.to_csv(
                    output_dir / f"attention_analysis_{args.baseline_key}_vs_{cond}.csv",
                    index=False,
                )

    # -----------------------------------------------------------------------
    # 3.  Cross-condition analysis (RQ4)
    # -----------------------------------------------------------------------
    if len(all_results) >= 2:
        logger.info("\nCross-condition analysis (RQ4) ...")
        cross = cross_condition_summary(all_results, alpha=args.alpha)
        cross.to_csv(output_dir / "cross_condition_summary.csv", index=False)

        shift_cols = [f"mean_shift_{c}" for c in all_results if f"mean_shift_{c}" in cross.columns]
        if shift_cols:
            plot_cross_condition_heatmap(cross, shift_cols, output_dir)

        # Asymmetry scatter (pejorative vs. laudatory)
        pej_key  = next((k for k in all_results if "pej" in k), None)
        laud_key = next((k for k in all_results if "laud" in k), None)
        asym_stats = None
        if pej_key and laud_key:
            plot_asymmetry_scatter(all_results[pej_key], all_results[laud_key], output_dir)
            asym_stats = asymmetry_stats(all_results[pej_key], all_results[laud_key])
            logger.info(f"  Asymmetry: r={asym_stats['pearson_r']}, "
                        f"{asym_stats['pct_opposite']:.1f}% opposite-direction codes")
    else:
        cross = pd.DataFrame()
        asym_stats = None

    # -----------------------------------------------------------------------
    # 4.  Report + Excel
    # -----------------------------------------------------------------------
    logger.info("\nBuilding report ...")
    report = build_report(
        all_results, cross, asym_stats,
        alpha=args.alpha,
        n_permutations=args.n_permutations,
        output_dir=output_dir,
    )
    print("\n" + report)

    build_excel_summary(all_results, output_dir)

    # -----------------------------------------------------------------------
    # 5.  Done
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info(f"Analysis complete. All outputs in: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
