# Attention Weight Calculation Review & Corrections

**Date:** December 2, 2025
**Reviewer:** Claude
**Status:** ⚠️ CRITICAL ISSUE FOUND - FIX REQUIRED

---

## Executive Summary

After comprehensive review of the attention weight extraction code in `prediction.py`, I've identified **one critical issue** and several recommendations for improving the scientific validity of attention analysis in this clinical valence testing study.

**Critical Issue:** Line 278 uses `aggregation="sum"` which artificially inflates attention weights for words with multiple sub-tokens, potentially biasing the analysis.

**Recommendation:** Change to `aggregation="average"` for accurate word-level attention analysis.

---

## Study Context

### What This Study Does

This project investigates **bias in clinical NLP models** by:

1. **Testing valence effects**: How subjective descriptors (pejorative: "difficult patient", laudatory: "cooperative patient", neutral: "typical patient") influence diagnostic predictions
2. **Analyzing attention patterns**: Which words receive model attention when making diagnosis predictions
3. **Measuring prediction shifts**: Comparing diagnosis probabilities before/after inserting valence terms

### Why Attention Weights Matter

Attention weights reveal:
- Which words the model focuses on for diagnosis
- Whether the model over-weights subjective descriptors
- If bias exists in clinical prediction models

**Example:**
```
Original text: "Patient presents with chest pain"
Shifted text: "Difficult patient presents with chest pain"

Question: Does the model now focus more on "difficult" than "chest pain"?
Answer: Attention analysis shows this.
```

---

## Attention Weight Extraction Analysis

### Current Implementation (`prediction.py` lines 92-134)

```python
def inference_from_texts(self, text: str, layer_num: int, head_num: int,
                        aggregation: str) -> Tuple[List[float], List[str], torch.Tensor]:
    # 1. Tokenize text
    inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(self.device)

    # 2. Run model with attention output
    outputs = self.model(**inputs, output_attentions=True)

    # 3. Extract attention for specific layer and head
    attentions = outputs.attentions[layer_num][0][head_num].cpu().numpy()

    # 4. Get CLS token attention to other tokens
    cls_attention = attentions[0, :][1:-1]  # ⚠️ REVIEW THIS

    # 5. Convert token IDs to words
    words = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[1:-1]

    # 6. Aggregate sub-tokens back to words
    final_attention_weights, input_words = [], []
    current_word, current_attention_sum, attention_count = "", 0.0, 0

    for i, token in enumerate(words):
        if token.startswith('##'):  # Sub-token continuation
            current_word += token[2:]
            if aggregation == "sum":
                current_attention_sum += cls_attention[i]  # ⚠️ ISSUE HERE
            # ... more aggregation logic
```

### Line 278: Critical Issue

```python
# In predict_batch method:
attention_weights, words, logits = super().inference_from_texts(
    sample,
    layer_num=self.layer_num,
    head_num=self.head_num,
    aggregation="sum"  # ❌ CRITICAL: Uses SUM instead of AVERAGE
)
```

---

## Issues Found

### 🔴 CRITICAL ISSUE 1: Sub-token Aggregation Bias

**Location:** `prediction.py:278` and `prediction.py:108-114`

**Problem:**
Using `aggregation="sum"` causes words with multiple sub-tokens to have artificially inflated attention scores.

**Example:**
```
Text: "The patient is playing"

Tokenization:
- "the" → ["the"] (1 token)
- "patient" → ["patient"] (1 token)
- "playing" → ["play", "##ing"] (2 tokens)

With SUM aggregation:
- "the": attention = 0.15
- "patient": attention = 0.35
- "playing": attention = 0.20 + 0.18 = 0.38  ← Artificially inflated!

Result: "playing" appears to have highest attention, but this is just because
it has more sub-tokens, not because the model actually focuses more on it!
```

**Impact on Study:**
- **Biased attention analysis**: Longer words appear more important
- **Invalid comparisons**: Can't compare attention across words of different lengths
- **Flawed conclusions**: May incorrectly identify which words influence predictions
- **Publication risk**: Reviewers will catch this methodological error

**Severity:** ⚠️ **CRITICAL** - Affects scientific validity of all attention analysis results

**Fix:**
```python
# Change line 278 from:
aggregation="sum"

# To:
aggregation="average"  # ✅ Normalizes by sub-token count
```

**Why AVERAGE is correct:**
```
With AVERAGE aggregation:
- "the": attention = 0.15 / 1 = 0.15
- "patient": attention = 0.35 / 1 = 0.35
- "playing": attention = (0.20 + 0.18) / 2 = 0.19  ← Normalized!

Now we can fairly compare which word has highest attention.
```

---

### 🟡 ISSUE 2: Attention Direction Not Documented

**Location:** `prediction.py:99`

**Code:**
```python
cls_attention = attentions[0, :][1:-1]
```

**Explanation:**
- `attentions[0, :]` = attention FROM [CLS] token TO all other tokens
- Alternative: `attentions[:, 0]` = attention TO [CLS] token FROM all other tokens

**Current behavior is CORRECT for classification tasks** because:
- [CLS] token is used for final prediction
- Attention FROM [CLS] shows what the classification head focuses on
- This is standard practice in BERT-based classification

**Recommendation:** Add documentation comment

```python
# Extract attention FROM [CLS] token TO each word token
# This shows what words the classification layer focuses on
cls_attention = attentions[0, :][1:-1]
```

**Status:** ✅ Currently correct, but should be documented

---

### 🟡 ISSUE 3: Missing Attention Normalization

**Location:** `prediction.py:99`

**Problem:**
Raw attention scores are used without normalization. For comparing across samples, normalized attention (sum to 1.0) is better practice.

**Current:**
```python
cls_attention = attentions[0, :][1:-1]  # Raw scores
```

**Recommended:**
```python
cls_attention = attentions[0, :][1:-1]
# Normalize to sum to 1.0 for fair comparison across samples
cls_attention = cls_attention / cls_attention.sum()
```

**Why this matters:**
- Different texts have different lengths
- Raw attention scores may not be directly comparable
- Normalized scores show *relative* importance within each sample

**Severity:** 🟡 **MEDIUM** - Affects comparability but doesn't invalidate results

---

### 🟢 ISSUE 4: Layer/Head Selection Documentation

**Location:** `prediction.py:142, 278`

**Current:** Uses layer 11, head 11 (last layer, last head)

**Status:** ✅ This is reasonable but somewhat arbitrary

**Recommendation:** Document why these defaults were chosen

```python
# Using last layer (11) and last head (11) as defaults
# Last layer typically captures high-level semantic information
# relevant for classification tasks. However, different layers/heads
# may capture different linguistic patterns. Users can customize
# via --layer_num and --head_num parameters.
```

**Note for publication:** Consider analyzing multiple layers/heads to show robustness

---

## What's Correct

### ✅ Correct Implementations

1. **[CLS] token usage** (Line 99)
   - Correctly uses first row `[0, :]` for classification attention
   - Appropriate for diagnosis prediction task

2. **Token exclusion** (Line 99, 100)
   - Correctly excludes [CLS] and [SEP] special tokens with `[1:-1]`
   - Only analyzes actual word tokens

3. **Sub-token identification** (Line 106)
   - Correctly identifies BERT word-pieces with `##` prefix
   - Proper handling of tokenization

4. **Batch handling** (Line 98)
   - Correctly extracts single sample with `[0]`
   - Appropriate for one-at-a-time inference

5. **Attention output** (Line 84, 96)
   - Correctly requests `output_attentions=True`
   - Properly accesses attention tensors

---

## Recommended Fixes

### Priority 1: CRITICAL FIX (Required for Publication)

**File:** `prediction.py`
**Line:** 278
**Change:**

```python
# BEFORE (INCORRECT):
attention_weights, words, logits = super().inference_from_texts(
    sample,
    layer_num=self.layer_num,
    head_num=self.head_num,
    aggregation="sum"  # ❌ WRONG
)

# AFTER (CORRECT):
attention_weights, words, logits = super().inference_from_texts(
    sample,
    layer_num=self.layer_num,
    head_num=self.head_num,
    aggregation="average"  # ✅ CORRECT
)
```

### Priority 2: Add Normalization (Recommended)

**File:** `prediction.py`
**Location:** After line 99
**Add:**

```python
cls_attention = attentions[0, :][1:-1]

# Normalize attention weights to sum to 1.0 for fair comparison
# This ensures attention scores are comparable across samples of different lengths
if cls_attention.sum() > 0:
    cls_attention = cls_attention / cls_attention.sum()
```

### Priority 3: Add Documentation (Recommended)

**File:** `prediction.py`
**Location:** Above `inference_from_texts` method
**Add comprehensive docstring:**

```python
def inference_from_texts(self, text: str, layer_num: int, head_num: int,
                        aggregation: str) -> Tuple[List[float], List[str], torch.Tensor]:
    """
    Extract attention weights and predictions for a clinical text sample.

    Attention Extraction Method:
    ----------------------------
    1. Extracts attention from the [CLS] token (used for classification)
    2. Uses specified transformer layer and attention head
    3. Aggregates sub-token attentions to word-level (BERT word-pieces)
    4. Returns normalized attention weights for interpretability

    Args:
        text: Clinical text to analyze
        layer_num: Transformer layer to extract attention from (0-11 for BioBERT)
                   Higher layers (9-11) capture semantic information
        head_num: Attention head to extract (0-11 for BioBERT)
                  Different heads focus on different linguistic patterns
        aggregation: How to aggregate sub-token attentions:
                    - "average": Mean attention (recommended - normalizes by token count)
                    - "sum": Sum attention (inflates multi-token words - not recommended)
                    - "max": Maximum attention across sub-tokens

    Returns:
        attention_weights: List of attention weights for each word
        words: List of words in the text (de-tokenized)
        logits: Model output logits for diagnosis prediction

    Note:
        - Uses attention FROM [CLS] TO other tokens (standard for classification)
        - Excludes special tokens ([CLS], [SEP]) from analysis
        - For publication-quality analysis, use aggregation="average"
    """
```

---

## Scientific Validity Assessment

### Before Fix:

- ❌ Attention weights biased by word length
- ❌ Cannot fairly compare attention across words
- ❌ Results may be misleading
- ❌ Would not pass peer review

### After Fix:

- ✅ Attention weights normalized by sub-token count
- ✅ Fair comparison across all words
- ✅ Scientifically valid attention analysis
- ✅ Publication-ready

---

## Testing Recommendations

After applying fixes, validate with these tests:

### Test 1: Sub-token Bias Check
```python
# Test text with words of different lengths
text = "The patient is playing cooperatively"

# Compare attention before/after fix
# Words like "cooperatively" (multiple sub-tokens) should not
# automatically have higher attention than short words
```

### Test 2: Attention Sum Verification
```python
# After normalization, attention should sum to ~1.0
attention_weights = [0.15, 0.35, 0.12, 0.19, 0.19]
assert abs(sum(attention_weights) - 1.0) < 0.01
```

### Test 3: Valence Word Attention
```python
# Test if valence words receive appropriate attention
text1 = "Patient presents with chest pain"
text2 = "Difficult patient presents with chest pain"

# After fix: Compare attention on "difficult" vs "chest pain"
# Should show relative importance, not just token count
```

---

## Impact on Existing Results

### If Previous Results Were Generated:

1. **Attention CSVs**: Need to be regenerated with corrected code
2. **Statistical Analysis**: May change if based on attention weights
3. **Visualizations**: Attention heatmaps/plots need updating
4. **Conclusions**: Review any conclusions about which words receive attention

### Action Items:

- [ ] Apply Priority 1 fix (aggregation="average")
- [ ] Re-run all experiments
- [ ] Regenerate all attention-based results
- [ ] Update visualizations
- [ ] Verify conclusions still hold
- [ ] Update manuscript/reports if needed

---

## Additional Recommendations

### For Publication:

1. **Report attention extraction method** in Methods section:
   ```
   "We extracted attention weights from the final transformer layer
   (layer 11) and final attention head (head 11) of the BioBERT model.
   Specifically, we analyzed attention from the [CLS] classification token
   to all word tokens, which indicates the importance of each word for the
   diagnosis prediction. For words tokenized into multiple sub-tokens by
   the BERT WordPiece tokenizer, we averaged attention weights across
   sub-tokens to obtain word-level attention scores. All attention weights
   were normalized to sum to 1.0 for comparability across samples."
   ```

2. **Acknowledge attention limitations**:
   ```
   "While attention weights provide insights into model behavior, they
   should be interpreted with caution as they may not fully explain model
   predictions (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019)."
   ```

3. **Robustness check**: Test multiple layers/heads to show results are robust

4. **Visualization**: Include attention heatmaps showing key examples

---

## References

**Attention Analysis in NLP:**
- Vaswani et al. (2017) - "Attention is All You Need"
- Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"
- Jain & Wallace (2019) - "Attention is not Explanation"
- Wiegreffe & Pinter (2019) - "Attention is not not Explanation"

**Clinical NLP:**
- van Aken et al. (2021) - "What Do You See in this Patient? Behavioral Testing of Clinical NLP Models"
- Lee et al. (2020) - "BioBERT: a pre-trained biomedical language representation model"

---

## Summary

### Critical Finding:
The use of `aggregation="sum"` in line 278 of `prediction.py` causes systematic bias in attention weight analysis, artificially inflating attention for longer words. This affects the scientific validity of any conclusions about which words influence model predictions.

### Required Action:
Change `aggregation="sum"` to `aggregation="average"` on line 278.

### Additional Recommendations:
1. Add attention normalization
2. Document attention extraction methodology
3. Re-run experiments with corrected code
4. Consider robustness checks across multiple layers/heads

### Status After Fix:
With the corrected aggregation method, the attention analysis will be scientifically sound and publication-ready.

---

**Prepared by:** AI Code Reviewer
**Date:** December 2, 2025
**Reviewed Files:** `prediction.py`, `valence_testing.py`, `main.py`
**Status:** ⚠️ Fix Required Before Publication
