# Approximate Randomization for Clinical Valence Testing
## Comprehensive Review and Implementation Guide

**Date:** 2025-12-17
**Author:** Statistical Analysis Review
**Reference:** Yeh, A. (2000). More Accurate Tests for the Statistical Significance of Result Differences. Proceedings of COLING.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Understanding Approximate Randomization](#understanding-approximate-randomization)
3. [Current Statistical Analysis Review](#current-statistical-analysis-review)
4. [R Code Implementation Analysis](#r-code-implementation-analysis)
5. [Application to Clinical Valence Testing](#application-to-clinical-valence-testing)
6. [Implementation Recommendations](#implementation-recommendations)
7. [Python Implementation Strategy](#python-implementation-strategy)
8. [Expected Benefits](#expected-benefits)

---

## 1. Executive Summary

### What is Approximate Randomization?

Approximate randomization (also called permutation testing or randomization testing) is a **non-parametric statistical method** that tests the significance of differences between two systems/methods without making distributional assumptions. Unlike traditional parametric tests (t-tests, ANOVA), it doesn't assume normality or equal variances.

### Key Insight from Yeh (2000)

The paper demonstrates that **traditional significance tests systematically underestimate p-values** when comparing NLP system outputs because they:
- Assume independence of samples (violated when same test set is used)
- Make parametric assumptions (normality, equal variance)
- Ignore the **stratification** of the evaluation data

### Why It Matters for Clinical Valence Testing

Your clinical valence testing framework compares:
- **Baseline predictions** (neutralized text)
- **Treatment predictions** (text with pejorative/laudatory/neutral valence)

These are **paired comparisons on the same clinical notes**, which violates independence assumptions. Approximate randomization provides **more accurate p-values** for these comparisons.

---

## 2. Understanding Approximate Randomization

### The Core Principle

**Question:** "Is the observed difference between Method A and Method B statistically significant?"

**Traditional approach:** Assume a distribution, calculate test statistic, look up p-value

**Randomization approach:**
1. Calculate the **observed test statistic** (e.g., difference in F1, accuracy, precision)
2. Generate the **null distribution** by randomly shuffling/permuting the labels
3. Count how many permutations produce a statistic **as extreme or more extreme** than observed
4. p-value = (# extreme permutations + 1) / (total permutations + 1)

### Mathematical Foundation

For two methods (New vs. Old) tested on *n* samples:

**Observed statistic:**
```
δ_observed = Metric(New) - Metric(Old)
```

**Null hypothesis:** There's no real difference between methods; observed difference is due to chance.

**Randomization procedure:**
For each sample *i*, we can assign the result to either "New" or "Old". Under the null hypothesis, all 2^n possible assignments are equally likely.

**P-value calculation:**
```
p = (count(|δ_permuted| ≥ |δ_observed|) + 1) / (R + 1)
```
where R = number of random permutations (typically R = 10,000)

### Stratification (Yeh's Key Contribution)

Yeh showed that when comparing methods on relation extraction tasks, you should **stratify by the confusion matrix cells**:

- Both methods find the relation (TP for both)
- Only Method I finds it
- Only Method II finds it
- Neither method finds it

This ensures that randomization respects the **structure** of the comparison.

---

## 3. Current Statistical Analysis Review

### What You're Currently Using

From `statistical_analysis.py`, your current approach includes:

#### 3.1 Parametric Tests
```python
def paired_ttest(self, baseline: np.ndarray, treatment: np.ndarray)
```
- **Assumption:** Differences are normally distributed
- **Limitation:** May not hold for probability differences or diagnosis predictions
- **Use case:** Good for large samples with approximately normal differences

#### 3.2 Non-Parametric Tests
```python
def wilcoxon_test(self, baseline: np.ndarray, treatment: np.ndarray)
```
- **Assumption:** Symmetric distribution of differences around median
- **Limitation:** Less powerful than randomization tests
- **Use case:** Good when normality violated but symmetry holds

#### 3.3 Bootstrap Confidence Intervals
```python
def bootstrap_confidence_interval(self, data: np.ndarray, ...)
```
- **Strength:** Non-parametric, no distributional assumptions
- **Limitation:** For CI estimation, not hypothesis testing

#### 3.4 Multiple Comparison Correction
```python
def correct_multiple_comparisons(self, p_values: np.ndarray)
```
- Uses FDR (Benjamini-Hochberg) and Bonferroni
- **Strength:** Controls family-wise error rate
- **Note:** Can be combined with randomization tests

### Current Gaps

1. **No paired randomization tests** for diagnosis probability shifts
2. **No stratified analysis** by prediction agreement patterns
3. **No sample-level permutation** for attention weight shifts
4. **Limited power** for small effect sizes with current parametric assumptions

---

## 4. R Code Implementation Analysis

### 4.1 The `evaluateSample` Function

**Purpose:** Calculate confusion matrix-based performance metrics (TP, FP, FN, Recall, Precision, F1, Accuracy)

**Key insight:**
```r
performanceMetrics <-
  data.frame( Method = c( newMethod , oldMethod ) ,
              TP = c( ... ) ,
              FP = c( ... ) ,
              FN = c( ... ) )
```

This creates a **contingency table** structure that enables stratified randomization.

### 4.2 Sample Structure

```r
getSampleSizeOfDisjunction <- function( sampleSet ){
  return( sampleSet %>%
    filter( Method %in% c( "I" , "II" ) ) %>%
    select( TrueRelations , SpuriousRelations ) %>%
    sum() )
}
```

The data structure has:
- **Method:** Which method(s) found the relation ("I", "II", "Both", "Neither")
- **TrueRelations:** Count of correct predictions
- **SpuriousRelations:** Count of false positives

### 4.3 Stratification Strategy

The key innovation is the **4-way stratification**:

| Method Category | Interpretation |
|----------------|----------------|
| "Both" | Both methods make the same prediction |
| "I" | Only Method I (new) makes this prediction |
| "II" | Only Method II (old) makes this prediction |
| "Neither" | Neither method makes this prediction |

**Why this matters:** When randomizing, you only swap within the disagreement categories ("I" vs "II"), preserving the agreement structure.

### 4.4 How Approximate Randomization Would Work

**Pseudo-code:**
```r
# 1. Calculate observed difference
observed_diff <- F1(newMethod) - F1(oldMethod)

# 2. Permutation loop
for (i in 1:10000) {
  # Randomly swap "I" and "II" labels (but keep "Both" and "Neither" fixed)
  permuted_sample <- randomize_within_strata(sampleSet)

  # Recalculate metrics
  permuted_metrics <- evaluateSample(permuted_sample)
  permuted_diff <- F1(newMethod) - F1(oldMethod)

  # Track if permuted difference is as extreme
  if (abs(permuted_diff) >= abs(observed_diff)) {
    count_extreme <- count_extreme + 1
  }
}

# 3. Calculate p-value
p_value <- (count_extreme + 1) / (10000 + 1)
```

---

## 5. Application to Clinical Valence Testing

### 5.1 Your Data Structure

For each clinical note, you have:

**Baseline (Neutralized):**
- Text: "Patient is alert and oriented."
- Predictions: [Prob(I10)=0.3, Prob(E11.9)=0.7, ...]

**Treatment (Pejorative):**
- Text: "Patient is difficult and non-compliant."
- Predictions: [Prob(I10)=0.4, Prob(E11.9)=0.6, ...]

**Comparison structure:**
```python
diagnosis_shifts = treatment_probs - baseline_probs
```

### 5.2 Mapping to Approximate Randomization

#### Current Comparison Pattern

You compare **paired** predictions for the same samples with different linguistic perturbations.

#### Stratification for Clinical Valence Testing

For each diagnosis code, you can stratify samples by **prediction agreement**:

| Stratum | Baseline Pred | Treatment Pred | Interpretation |
|---------|---------------|----------------|----------------|
| TP-TP | Positive (≥0.5) | Positive (≥0.5) | Both predict this diagnosis |
| TP-FN | Positive | Negative (<0.5) | Only baseline predicts |
| FN-TP | Negative | Positive | Only treatment predicts |
| TN-TN | Negative | Negative | Neither predicts |

**Key insight:** Under the null hypothesis (valence has no effect), swapping baseline ↔ treatment labels within disagreement strata should produce similar metric differences.

### 5.3 Sample-Level Randomization

For **continuous metrics** (probability differences, attention weights):

**Null hypothesis:** Valence shift has no effect on predictions.

**Randomization strategy:**
For each sample *i*, randomly decide whether to:
- Keep: `diff[i] = treatment[i] - baseline[i]`
- Flip: `diff[i] = -(treatment[i] - baseline[i])` = `baseline[i] - treatment[i]`

This is equivalent to randomly swapping the labels "baseline" and "treatment" for each sample.

**Test statistic:** Mean difference
```python
observed_mean_diff = np.mean(treatment_probs - baseline_probs)
```

**Permutation:**
```python
for permutation in range(R):
    # Randomly flip signs
    flip_mask = np.random.choice([1, -1], size=n_samples)
    permuted_diffs = (treatment_probs - baseline_probs) * flip_mask
    permuted_mean = np.mean(permuted_diffs)

    if abs(permuted_mean) >= abs(observed_mean_diff):
        count_extreme += 1

p_value = (count_extreme + 1) / (R + 1)
```

### 5.4 Multi-Metric Randomization

For comprehensive evaluation, you can test:

1. **Diagnosis-level metrics:**
   - Probability shifts (mean, median)
   - Classification agreement (accuracy, F1, precision, recall)

2. **Attention-level metrics:**
   - Mean attention weight changes
   - Top-k attention word differences

3. **Aggregate metrics:**
   - Overall model performance degradation
   - Number of diagnosis changes per sample

**Advantage:** All metrics use the **same set of permutations**, ensuring consistency and reducing computation.

---

## 6. Implementation Recommendations

### 6.1 Priority 1: Paired Permutation Test for Diagnosis Shifts

**Goal:** Test if valence shifts significantly affect diagnosis predictions.

**Implementation:**
```python
def paired_permutation_test(
    self,
    baseline: np.ndarray,
    treatment: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    random_seed: Optional[int] = None
) -> StatisticalTestResult:
    """
    Perform paired permutation test (exact for small n, approximate for large n).

    Args:
        baseline: Baseline group observations
        treatment: Treatment group observations
        n_permutations: Number of random permutations
        alternative: 'two-sided', 'greater', or 'less'
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

    for _ in range(n_permutations):
        # Randomly flip signs (equivalent to swapping labels)
        flip_mask = np.random.choice([-1, 1], size=n_samples)
        permuted_diff = np.mean((treatment - baseline) * flip_mask)

        # Count extreme values based on alternative hypothesis
        if alternative == "two-sided":
            if abs(permuted_diff) >= abs(observed_diff):
                count_extreme += 1
        elif alternative == "greater":
            if permuted_diff >= observed_diff:
                count_extreme += 1
        elif alternative == "less":
            if permuted_diff <= observed_diff:
                count_extreme += 1

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
            'mean_difference': observed_diff
        }
    )
```

### 6.2 Priority 2: Stratified Randomization for Binary Outcomes

**Goal:** Test diagnosis classification changes with proper stratification.

**Implementation:**
```python
def stratified_permutation_test(
    self,
    baseline_binary: np.ndarray,
    treatment_binary: np.ndarray,
    metric_func: callable,
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

    Args:
        baseline_binary: Binary predictions for baseline
        treatment_binary: Binary predictions for treatment
        metric_func: Function to compute metric (e.g., accuracy, F1)
        n_permutations: Number of permutations
        random_seed: Random seed

    Returns:
        StatisticalTestResult
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Identify strata
    both_positive = (baseline_binary == 1) & (treatment_binary == 1)
    both_negative = (baseline_binary == 0) & (treatment_binary == 0)
    baseline_only = (baseline_binary == 1) & (treatment_binary == 0)
    treatment_only = (baseline_binary == 0) & (treatment_binary == 1)

    # Calculate observed metric
    observed_metric = metric_func(baseline_binary, treatment_binary)

    count_extreme = 0

    for _ in range(n_permutations):
        # Create permuted arrays (start with originals)
        perm_baseline = baseline_binary.copy()
        perm_treatment = treatment_binary.copy()

        # Only randomize within disagreement strata
        n_disagree = np.sum(baseline_only) + np.sum(treatment_only)
        if n_disagree > 0:
            # Randomly assign disagreements to baseline_only or treatment_only
            random_assignment = np.random.permutation(n_disagree)
            n_baseline_only = np.sum(baseline_only)

            # Get indices
            disagree_indices = np.where(baseline_only | treatment_only)[0]

            # Assign first n_baseline_only to baseline_only, rest to treatment_only
            new_baseline_only_idx = disagree_indices[random_assignment < n_baseline_only]
            new_treatment_only_idx = disagree_indices[random_assignment >= n_baseline_only]

            # Reset disagreement positions
            perm_baseline[baseline_only | treatment_only] = 0
            perm_treatment[baseline_only | treatment_only] = 0

            # Set new assignments
            perm_baseline[new_baseline_only_idx] = 1
            perm_treatment[new_treatment_only_idx] = 1

        # Preserve agreement strata
        perm_baseline[both_positive] = 1
        perm_treatment[both_positive] = 1
        perm_baseline[both_negative] = 0
        perm_treatment[both_negative] = 0

        # Calculate permuted metric
        permuted_metric = metric_func(perm_baseline, perm_treatment)

        if abs(permuted_metric - 0) >= abs(observed_metric - 0):
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
            'n_treatment_only': int(np.sum(treatment_only))
        }
    )
```

### 6.3 Priority 3: Multi-Metric Permutation Framework

**Goal:** Efficiently compute permutation tests for multiple metrics simultaneously.

**Implementation:**
```python
def multi_metric_permutation_test(
    self,
    baseline: np.ndarray,
    treatment: np.ndarray,
    metric_functions: Dict[str, callable],
    n_permutations: int = 10000,
    random_seed: Optional[int] = None
) -> Dict[str, StatisticalTestResult]:
    """
    Perform permutation tests for multiple metrics simultaneously.

    Args:
        baseline: Baseline observations
        treatment: Treatment observations
        metric_functions: Dict mapping metric names to functions
        n_permutations: Number of permutations
        random_seed: Random seed

    Returns:
        Dictionary mapping metric names to StatisticalTestResult
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate observed metrics
    observed_metrics = {
        name: func(baseline, treatment)
        for name, func in metric_functions.items()
    }

    # Initialize counters
    count_extreme = {name: 0 for name in metric_functions.keys()}

    # Generate permutations (reuse for all metrics)
    for _ in range(n_permutations):
        flip_mask = np.random.choice([-1, 1], size=len(baseline))
        permuted_baseline = np.where(flip_mask == 1, baseline, treatment)
        permuted_treatment = np.where(flip_mask == 1, treatment, baseline)

        # Calculate all metrics for this permutation
        for name, func in metric_functions.items():
            permuted_metric = func(permuted_baseline, permuted_treatment)
            if abs(permuted_metric) >= abs(observed_metrics[name]):
                count_extreme[name] += 1

    # Create results
    results = {}
    for name in metric_functions.keys():
        p_value = (count_extreme[name] + 1) / (n_permutations + 1)
        results[name] = StatisticalTestResult(
            test_name=f"Permutation Test ({name})",
            statistic=observed_metrics[name],
            p_value=p_value,
            significant=p_value < self.significance_level,
            additional_info={'n_permutations': n_permutations}
        )

    return results
```

### 6.4 Integration into `analyze_diagnosis_shifts`

**Modify the existing function to include permutation tests:**

```python
def analyze_diagnosis_shifts(
    self,
    baseline_probs: pd.DataFrame,
    treatment_probs: pd.DataFrame,
    diagnosis_codes: List[str],
    use_permutation: bool = True,
    n_permutations: int = 10000
) -> pd.DataFrame:
    """
    Comprehensive analysis with permutation tests.
    """
    results = []

    for code in diagnosis_codes:
        baseline = baseline_probs[code].values
        treatment = treatment_probs[code].values

        # Existing tests
        ttest_result = self.paired_ttest(baseline, treatment)
        wilcoxon_result = self.wilcoxon_test(baseline, treatment)

        # NEW: Permutation test
        if use_permutation:
            perm_result = self.paired_permutation_test(
                baseline, treatment,
                n_permutations=n_permutations
            )

        results.append({
            'diagnosis_code': code,
            # ... existing fields ...
            'permutation_pvalue': perm_result.p_value if use_permutation else None,
            'permutation_significant': perm_result.significant if use_permutation else None,
            # ... other fields ...
        })

    results_df = pd.DataFrame(results)

    # Apply multiple comparison correction to permutation p-values
    if use_permutation and len(results_df) > 1:
        rejected_perm, corrected_p_perm = self.correct_multiple_comparisons(
            results_df['permutation_pvalue'].values
        )
        results_df['permutation_pvalue_corrected'] = corrected_p_perm
        results_df['permutation_significant_corrected'] = rejected_perm

    return results_df
```

---

## 7. Python Implementation Strategy

### 7.1 New Class: `PermutationTester`

Create a dedicated class for permutation-based inference:

```python
class PermutationTester:
    """
    Permutation-based statistical inference for paired comparisons.
    """

    def __init__(
        self,
        n_permutations: int = 10000,
        significance_level: float = 0.05,
        random_seed: Optional[int] = None,
        n_jobs: int = 1
    ):
        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.random_seed = random_seed
        self.n_jobs = n_jobs  # For parallel computation

    def paired_test(self, baseline, treatment, statistic='mean'):
        """Paired permutation test."""
        pass

    def stratified_test(self, baseline_binary, treatment_binary, metric):
        """Stratified permutation test for binary outcomes."""
        pass

    def bootstrap_permutation_ci(self, baseline, treatment):
        """Combined bootstrap-permutation for CI + p-value."""
        pass
```

### 7.2 Integration Points

**File: `statistical_analysis.py`**

1. Add `PermutationTester` class
2. Integrate into `StatisticalAnalyzer.analyze_diagnosis_shifts()`
3. Add to `StatisticalAnalyzer.analyze_attention_shifts()`
4. Update `generate_analysis_report()` to include permutation results

**File: `valence_testing.py`**

1. Add option to enable permutation testing in `run_test()`
2. Pass configuration to statistical analyzer

**File: `config.yaml`**

```yaml
statistical_analysis:
  significance_level: 0.05
  correction_method: "fdr_bh"
  use_permutation_tests: true
  n_permutations: 10000
  permutation_seed: 42
```

### 7.3 Computational Optimization

**Challenge:** 10,000 permutations × 187 diagnosis codes × multiple shifts = expensive

**Solutions:**

1. **Parallel processing:**
```python
from joblib import Parallel, delayed

def permutation_test_parallel(baseline, treatment, n_permutations, n_jobs=-1):
    """Run permutations in parallel."""

    observed_diff = np.mean(treatment - baseline)

    def single_permutation(seed):
        np.random.seed(seed)
        flip_mask = np.random.choice([-1, 1], size=len(baseline))
        permuted_diff = np.mean((treatment - baseline) * flip_mask)
        return abs(permuted_diff) >= abs(observed_diff)

    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_permutation)(seed)
        for seed in range(n_permutations)
    )

    count_extreme = sum(results)
    p_value = (count_extreme + 1) / (n_permutations + 1)

    return p_value
```

2. **Adaptive permutations:**
```python
def adaptive_permutation_test(baseline, treatment, max_permutations=10000, min_permutations=1000):
    """
    Stop early if p-value estimate is stable.
    """
    observed_diff = np.mean(treatment - baseline)
    count_extreme = 0

    for i in range(max_permutations):
        flip_mask = np.random.choice([-1, 1], size=len(baseline))
        permuted_diff = np.mean((treatment - baseline) * flip_mask)

        if abs(permuted_diff) >= abs(observed_diff):
            count_extreme += 1

        # Check for early stopping (after minimum permutations)
        if i >= min_permutations and i % 100 == 0:
            p_current = (count_extreme + 1) / (i + 1)

            # If p-value is very significant or very non-significant, can stop early
            if p_current < 0.001 or p_current > 0.1:
                return p_current

    return (count_extreme + 1) / (max_permutations + 1)
```

3. **Vectorized permutations:**
```python
def vectorized_permutation_test(baseline, treatment, n_permutations=10000):
    """
    Generate all permutations at once (memory intensive but fast).
    """
    n_samples = len(baseline)
    observed_diff = np.mean(treatment - baseline)

    # Generate all flip masks at once
    flip_masks = np.random.choice([-1, 1], size=(n_permutations, n_samples))

    # Vectorized difference calculation
    diffs = treatment - baseline
    permuted_diffs = flip_masks * diffs[np.newaxis, :]
    permuted_means = np.mean(permuted_diffs, axis=1)

    count_extreme = np.sum(np.abs(permuted_means) >= abs(observed_diff))
    p_value = (count_extreme + 1) / (n_permutations + 1)

    return p_value
```

---

## 8. Expected Benefits

### 8.1 Statistical Rigor

1. **Exact p-values under minimal assumptions:**
   - No normality assumption
   - No variance homogeneity assumption
   - Valid for small samples

2. **Proper handling of paired structure:**
   - Respects within-sample correlation
   - Accounts for repeated measures design

3. **Stratification preserves comparison structure:**
   - More powerful than unstratified tests
   - Mirrors Yeh's (2000) methodology for NLP evaluation

### 8.2 Publication Quality

1. **Aligns with best practices in NLP evaluation:**
   - Yeh (2000) is well-cited in computational linguistics
   - Shows awareness of statistical issues in paired comparisons

2. **Robustness checks:**
   - Can report both parametric (t-test) and non-parametric (permutation) results
   - Convergence of results increases confidence

3. **Reviewer credibility:**
   - Demonstrates statistical sophistication
   - Addresses potential reviewer concerns about p-value validity

### 8.3 Practical Advantages

1. **Effect size interpretation unchanged:**
   - Still use Cohen's d, Hedges' g
   - Permutation provides p-value, not effect size

2. **Multiple comparison correction still applicable:**
   - FDR, Bonferroni work with permutation p-values
   - Can control family-wise error rate across diagnoses

3. **Flexible framework:**
   - Easy to adapt to new metrics (custom functions)
   - Can test interaction effects, subgroup differences

### 8.4 Computational Feasibility

**Estimated runtime:**
- 187 diagnoses × 10,000 permutations = 1.87M calculations
- With vectorization: ~1-2 minutes per shift comparison
- With parallelization (8 cores): ~15-30 seconds per shift

**Total analysis time:** ~2-5 minutes (acceptable for publication-quality analysis)

---

## 9. Comparison Table: Current vs. Approximate Randomization

| Aspect | Current Approach | With Approximate Randomization |
|--------|------------------|-------------------------------|
| **Primary test** | Paired t-test | Paired permutation test |
| **Assumptions** | Normality of differences | Exchangeability under H₀ (minimal) |
| **P-value accuracy** | Approximate (assumes distribution) | Exact (for finite permutations) |
| **Small sample performance** | Poor (n < 30) | Good (valid for any n) |
| **Stratification** | Not used | Used (by agreement pattern) |
| **Computational cost** | O(n) - instant | O(n × R) - seconds to minutes |
| **Effect size** | Cohen's d | Cohen's d (unchanged) |
| **Multiple testing** | FDR correction | FDR correction (same) |
| **Publication acceptance** | Standard | Gold standard for NLP |

---

## 10. Next Steps

### Phase 1: Core Implementation (Priority: HIGH)
- [ ] Add `paired_permutation_test()` to `StatisticalAnalyzer`
- [ ] Integrate into `analyze_diagnosis_shifts()`
- [ ] Add configuration options to `config.yaml`
- [ ] Write unit tests with known distributions

### Phase 2: Advanced Features (Priority: MEDIUM)
- [ ] Implement `stratified_permutation_test()`
- [ ] Add `multi_metric_permutation_test()`
- [ ] Implement parallel processing for speed
- [ ] Add adaptive permutation stopping

### Phase 3: Validation (Priority: HIGH)
- [ ] Compare permutation p-values with t-test p-values
- [ ] Validate on synthetic data with known null
- [ ] Create visualization comparing test results
- [ ] Document when results diverge (heavy tails, outliers)

### Phase 4: Documentation (Priority: MEDIUM)
- [ ] Update README with permutation testing explanation
- [ ] Add section to statistical analysis report
- [ ] Create tutorial notebook demonstrating usage
- [ ] Write methods section for publication

---

## 11. References

1. **Yeh, A. (2000).** "More Accurate Tests for the Statistical Significance of Result Differences." *Proceedings of COLING*, 947-953.

2. **Good, P. I. (2013).** *Permutation Tests: A Practical Guide to Resampling Methods for Testing Hypotheses*. Springer Science & Business Media.

3. **Noreen, E. W. (1989).** *Computer-Intensive Methods for Testing Hypotheses*. John Wiley & Sons.

4. **Dror, R., et al. (2018).** "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing." *Proceedings of ACL*, 1383-1392.

5. **Berg-Kirkpatrick, T., et al. (2012).** "An Empirical Investigation of Statistical Significance in NLP." *Proceedings of EMNLP*, 995-1005.

---

## 12. Conclusion

Approximate randomization testing, as described by Yeh (2000), provides a **statistically rigorous framework** for evaluating the significance of prediction differences in your clinical valence testing project. By implementing paired permutation tests with proper stratification, you will:

1. **Improve statistical validity** by avoiding unwarranted parametric assumptions
2. **Align with NLP best practices** for system comparison
3. **Increase publication credibility** through methodological rigor
4. **Enable more powerful detection** of valence effects through stratification

The implementation is straightforward, computationally feasible, and integrates cleanly with your existing statistical analysis pipeline. The benefits significantly outweigh the modest development effort required.

**Recommendation:** Implement paired permutation testing as the **primary** significance test, while retaining parametric tests for **comparison and robustness checks**. Report both in your analysis, noting convergence or divergence of results.

---

**End of Document**
