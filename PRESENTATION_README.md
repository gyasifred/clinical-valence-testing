# Research Presentation Document

## Overview

This directory contains a comprehensive technical presentation document (`RESEARCH_PRESENTATION.tex`) that explains the Clinical Valence Testing research project. The document is designed for presentation to technical Principal Investigators (PIs) and includes:

- **Detailed methodology**
- **Experimental design**
- **Expected analyses** (with placeholder values)
- **Visualizations** (publication-quality figures)
- **Statistical framework**
- **Implementation details**

**Format:** LaTeX document that compiles to PDF
**Length:** ~30 pages
**Target Audience:** Technical PIs, research reviewers, grant committees

---

## Document Contents

### 1. Introduction and Motivation (Pages 1-3)
- Background on clinical NLP and bias
- Research question and significance
- Clinical implications

### 2. Research Objectives (Page 4)
- Primary objectives (quantify bias, analyze attention)
- Secondary objectives (diagnosis sensitivity, robustness)

### 3. Methodology (Pages 5-10)
- Model architecture (BioBERT)
- Experimental design (shift transformations)
- Data processing pipeline
- Attention weight extraction (mathematical formulation)

### 4. Experimental Setup (Pages 11-13)
- Dataset description
- Hyperparameters
- Evaluation metrics
- Statistical tests

### 5. Expected Analyses (Pages 14-18)
- Diagnosis probability shifts (bar charts)
- Attention weight distributions (box plots)
- Statistical hypotheses
- Attention heatmaps

### 6. Technical Implementation (Pages 19-20)
- Software architecture
- Reproducibility measures
- Computational requirements

### 7. Quality Assurance (Pages 21-22)
- Validation procedures
- Code quality standards

### 8. Limitations and Considerations (Page 23)
- Methodological limitations
- Dataset considerations
- Interpretation caveats

### 9. Expected Impact (Pages 24-25)
- Scientific contributions
- Clinical implications
- Regulatory relevance

### 10. Timeline, Dissemination, Ethics (Pages 26-28)
- Research timeline
- Publication plan
- Ethical considerations

### 11. Appendices (Pages 29-30)
- Valence term dictionaries
- Sample output formats
- Statistical analysis pseudocode

---

## Compilation Instructions

### Prerequisites

Install LaTeX distribution:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
- Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

### Quick Compilation

**Option 1: Use the provided script (Linux/macOS)**
```bash
./compile_presentation.sh
```

**Option 2: Manual compilation**
```bash
# Compile three times for proper references
pdflatex RESEARCH_PRESENTATION.tex
pdflatex RESEARCH_PRESENTATION.tex
pdflatex RESEARCH_PRESENTATION.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc *.bbl *.blg
```

**Option 3: Using latexmk (if installed)**
```bash
latexmk -pdf RESEARCH_PRESENTATION.tex
latexmk -c  # Clean up
```

### Output

After successful compilation:
- **PDF file:** `RESEARCH_PRESENTATION.pdf`
- **Size:** ~2-3 MB
- **Pages:** ~30 pages

---

## Viewing the PDF

**Linux:**
```bash
xdg-open RESEARCH_PRESENTATION.pdf
```

**macOS:**
```bash
open RESEARCH_PRESENTATION.pdf
```

**Windows:**
```bash
start RESEARCH_PRESENTATION.pdf
```

---

## Key Features

### 1. Publication-Quality Visualizations

All figures are generated using TikZ/PGFPlots:
- ✅ Vector graphics (scales perfectly)
- ✅ Consistent styling
- ✅ Color-coded for valence types
- ✅ Professional appearance

### 2. Placeholder Data

All results use placeholder values (0.XX, X.XX):
- ✅ **No fabricated results**
- ✅ Shows expected structure
- ✅ Easy to update with real data
- ✅ Maintains scientific integrity

### 3. Mathematical Rigor

Includes formal mathematical notation for:
- Attention weight extraction
- Normalization procedures
- Statistical tests
- Effect size calculations

### 4. Comprehensive Coverage

Document covers:
- ✅ Research design
- ✅ Technical implementation
- ✅ Quality assurance
- ✅ Ethical considerations
- ✅ Expected impact
- ✅ Timeline and dissemination

---

## Customization

### Updating Placeholder Values

To insert real results, search and replace in the `.tex` file:

```bash
# Replace 0.XX with actual values
sed -i 's/0\.XX/0.123/g' RESEARCH_PRESENTATION.tex

# Replace X.XX with actual values
sed -i 's/X\.XX/5.67/g' RESEARCH_PRESENTATION.tex
```

### Adding Your Team Information

Edit lines 23-27 in `RESEARCH_PRESENTATION.tex`:

```latex
\author{
Your Institution Name \\
Research Team Members \\
\texttt{your-email@institution.edu}
}
```

### Modifying Visualizations

All figures are generated with TikZ code that can be customized:

**Example: Update bar chart data**
```latex
% Find coordinates section in the figure
\addplot[fill=pejorative, draw=black] coordinates {
    (F32.9, 0.XX)  % Replace 0.XX with real value
    (F41.9, 0.XX)  % Replace 0.XX with real value
    ...
};
```

### Adding Sections

To add new sections:
```latex
\section{Your New Section}
\subsection{Your Subsection}

Your content here...
```

---

## Troubleshooting

### Issue 1: "pdflatex: command not found"

**Solution:** Install LaTeX (see Prerequisites above)

### Issue 2: "Missing package" errors

**Solution:** Install full TeX Live distribution:
```bash
sudo apt-get install texlive-full  # Ubuntu/Debian
```

### Issue 3: Figures not appearing

**Solution:** Ensure these packages are available:
- `tikz`
- `pgfplots`
- Compile three times to generate all figures

### Issue 4: Bibliography errors

**Solution:** The document uses embedded bibliography (no separate .bib file needed)

### Issue 5: Compilation takes long time

**Expected:** First compilation may take 1-2 minutes due to complex TikZ figures

---

## Document Structure

```
RESEARCH_PRESENTATION.tex
├── Preamble (packages and settings)
├── Title and abstract
├── Table of contents
├── Main content (sections 1-11)
├── Bibliography
└── Appendices
```

### LaTeX Packages Used

| Package | Purpose |
|---------|---------|
| `geometry` | Page margins |
| `graphicx` | Image handling |
| `amsmath`, `amssymb` | Mathematical notation |
| `booktabs` | Professional tables |
| `hyperref` | Clickable links |
| `tikz`, `pgfplots` | Visualizations |
| `xcolor` | Color definitions |
| `natbib` | Bibliography |

---

## Tips for Presentation

### 1. Print Version

For printed copies, consider:
```latex
% Add before \begin{document}
\usepackage[colorlinks=false]{hyperref}  % Remove colored links
```

### 2. Slides Version

To create slides from this content:
```latex
\documentclass{beamer}
% Extract key sections
% Use more concise text
% Enlarge figures
```

### 3. Two-Column Format

For journal submission:
```latex
\documentclass[twocolumn]{article}
```

### 4. Supplementary Materials

Export appendices separately:
```latex
% Compile only appendices
\includeonly{appendix}
```

---

## Integration with Results

### When You Have Real Data

1. **Export results** from your experiments to CSV
2. **Create data scripts** to generate LaTeX tables:

```python
import pandas as pd

# Load results
df = pd.read_csv('results.csv')

# Generate LaTeX table
latex_table = df.to_latex(index=False)

# Insert into document
print(latex_table)
```

3. **Update figures** with actual values
4. **Recompile** the document

### Automated Updates

Create a script to inject results:

```python
#!/usr/bin/env python3
import re

# Read template
with open('RESEARCH_PRESENTATION.tex', 'r') as f:
    content = f.read()

# Load your results
results = load_experimental_results()

# Replace placeholders
for key, value in results.items():
    pattern = f'PLACEHOLDER_{key}'
    content = re.sub(pattern, str(value), content)

# Write updated document
with open('RESEARCH_PRESENTATION_UPDATED.tex', 'w') as f:
    f.write(content)
```

---

## Version Control

This document is version-controlled with your research code:

```bash
# Track changes
git add RESEARCH_PRESENTATION.tex
git commit -m "Update presentation with methodology details"

# View history
git log RESEARCH_PRESENTATION.tex

# Compare versions
git diff HEAD~1 RESEARCH_PRESENTATION.tex
```

---

## Sharing the Document

### For Review

**PDF only:**
```bash
# Share the compiled PDF
scp RESEARCH_PRESENTATION.pdf reviewer@server:~/
```

**With source:**
```bash
# Create archive with source and PDF
tar -czf presentation.tar.gz \
    RESEARCH_PRESENTATION.tex \
    RESEARCH_PRESENTATION.pdf \
    compile_presentation.sh
```

### For Publication

When submitting to journals:
1. Include `.tex` source file
2. Include `.pdf` compiled version
3. Include any external figures (if added)
4. Specify compilation instructions

---

## FAQ

**Q: Can I edit in Overleaf?**
A: Yes! Upload `RESEARCH_PRESENTATION.tex` to [Overleaf](https://www.overleaf.com/) for online editing

**Q: How do I add my institution logo?**
A: Add before `\maketitle`:
```latex
\titlegraphic{\includegraphics[width=3cm]{logo.png}}
```

**Q: Can I change the color scheme?**
A: Yes, modify the color definitions:
```latex
\definecolor{clinicalblue}{RGB}{0,102,204}  % Change RGB values
```

**Q: How do I export just one section?**
A: Use `\includeonly`:
```latex
\includeonly{section3}  % Only compile section 3
```

**Q: Can I convert to Word format?**
A: Use pandoc:
```bash
pandoc RESEARCH_PRESENTATION.tex -o output.docx
```

**Q: How do I add more references?**
A: Add to bibliography section:
```latex
\bibitem{yourkey}
Author, A. (Year).
\newblock Title.
\newblock \textit{Journal}, volume(issue), pages.
```

---

## Support

For LaTeX help:
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [TeX Stack Exchange](https://tex.stackexchange.com/)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)

For TikZ/PGFPlots:
- [PGFPlots Manual](http://pgfplots.sourceforge.net/pgfplots.pdf)
- [TikZ Examples](http://www.texample.net/tikz/)

---

## Summary

This research presentation document provides:

✅ **Comprehensive coverage** of your clinical valence testing research
✅ **Publication-quality** formatting and visualizations
✅ **Placeholder values** maintaining scientific integrity
✅ **Easy compilation** with provided scripts
✅ **Professional appearance** suitable for PIs and reviewers
✅ **Customizable** for your specific needs

**Ready to compile:** `./compile_presentation.sh`

**Next steps:**
1. Review the document structure
2. Customize team/institution information
3. Compile to PDF
4. Present to PIs or reviewers
5. Update with real results when available

---

**Document Version:** 1.0
**Last Updated:** December 2, 2025
**Status:** Ready for compilation
