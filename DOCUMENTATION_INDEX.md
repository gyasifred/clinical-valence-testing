# Documentation Index

Complete documentation for the Clinical Valence Testing framework.

## 📚 Documentation Overview

This project includes comprehensive documentation for users at all levels, from beginners to advanced researchers.

---

## 🚀 Getting Started

### New Users Start Here

1. **[README.md](README.md)** - **START HERE**
   - Quick installation guide
   - Minimal working example
   - Basic command reference
   - **Time to read:** 10 minutes
   - **Who should read:** Everyone

2. **[INSTALLATION.md](#installation)** - Detailed Installation
   - System requirements
   - Platform-specific instructions
   - Troubleshooting installation issues
   - **Time to read:** 5 minutes
   - **Who should read:** First-time users

---

## 📖 Core Documentation

### For Regular Usage

3. **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Comprehensive Usage Guide
   - Step-by-step workflows
   - Common use cases with examples
   - Result interpretation
   - Best practices
   - **Time to read:** 30 minutes
   - **Who should read:** All users running experiments

4. **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Configuration Reference
   - Complete parameter documentation
   - Configuration examples
   - Command-line override rules
   - Performance tuning
   - **Time to read:** 20 minutes
   - **Who should read:** Users customizing experiments

---

## 🔬 Technical Documentation

### For Understanding the System

5. **[PROJECT_ANALYSIS_REPORT.md](PROJECT_ANALYSIS_REPORT.md)** - System Architecture
   - Project structure
   - Component descriptions
   - Data flow
   - Technical design decisions
   - **Time to read:** 15 minutes
   - **Who should read:** Developers, researchers

6. **[CODE_REVIEW.md](CODE_REVIEW.md)** - Code Quality Review
   - Known issues and fixes
   - Implementation decisions
   - Scientific validity notes
   - **Time to read:** 10 minutes
   - **Who should read:** Contributors, reviewers

7. **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - Enhancement History
   - Refactoring details
   - New features added
   - Migration guide
   - **Time to read:** 10 minutes
   - **Who should read:** Users upgrading from old versions

8. **[REFACTORING_PLAN.md](REFACTORING_PLAN.md)** - Development Roadmap
   - Planned improvements
   - Phased implementation
   - Future features
   - **Time to read:** 5 minutes
   - **Who should read:** Contributors, maintainers

---

## 📋 Quick Reference

### Cheat Sheets

**Quick Start Commands:**
```bash
# Install
pip install -r requirements.txt

# Run all shifts
python main.py --test_set_path ./data/test.csv --model_path MODEL_NAME

# Run specific shift
python main.py --test_set_path ./data/test.csv --shift_keys pejorative

# With GPU
python main.py --test_set_path ./data/test.csv --gpu true --batch_size 256

# Debug mode
python main.py --test_set_path ./data/test.csv --config_path config_debug.yaml
```

**Common Config Patterns:**
```yaml
# Quick test
model:
  batch_size: 256
data:
  max_samples: 100
analysis:
  run_statistical: false

# Publication quality
model:
  batch_size: 128
analysis:
  correction_method: "bonferroni"
  bootstrap_iterations: 10000
random_seed: 42
deterministic: true

# Large dataset
model:
  batch_size: 64
prediction:
  checkpoint_interval: 5000
  save_attention: false
output:
  compression: "gzip"
```

---

## 🎯 Documentation by User Type

### For First-Time Users
**Read these in order:**
1. [README.md](README.md) - Installation and quick start
2. [USAGE_GUIDE.md](USAGE_GUIDE.md) - Basic workflows
3. [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - Customization

**Goal:** Run your first experiment successfully

### For Researchers
**Read these:**
1. [README.md](README.md) - Quick reference
2. [USAGE_GUIDE.md](USAGE_GUIDE.md) - Use cases and interpretation
3. [PROJECT_ANALYSIS_REPORT.md](PROJECT_ANALYSIS_REPORT.md) - Understand the science
4. [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - Fine-tune experiments

**Goal:** Conduct rigorous experiments for publication

### For Developers
**Read these:**
1. [PROJECT_ANALYSIS_REPORT.md](PROJECT_ANALYSIS_REPORT.md) - Architecture
2. [CODE_REVIEW.md](CODE_REVIEW.md) - Code quality notes
3. [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) - Recent changes
4. [USAGE_GUIDE.md](USAGE_GUIDE.md) - Advanced scenarios

**Goal:** Understand and extend the codebase

### For Code Reviewers
**Read these:**
1. [PROJECT_ANALYSIS_REPORT.md](PROJECT_ANALYSIS_REPORT.md) - System overview
2. [CODE_REVIEW.md](CODE_REVIEW.md) - Known issues
3. [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - Development plan

**Goal:** Evaluate code quality and design

---

## 📊 Documentation by Topic

### Installation & Setup
- [README.md § Installation](README.md#installation)
- [README.md § Quick Start](README.md#quick-start)
- [USAGE_GUIDE.md § Getting Started](USAGE_GUIDE.md#getting-started)

### Running Experiments
- [README.md § Step-by-Step Usage Guide](README.md#step-by-step-usage-guide)
- [USAGE_GUIDE.md § Basic Workflows](USAGE_GUIDE.md#basic-workflows)
- [USAGE_GUIDE.md § Common Use Cases](USAGE_GUIDE.md#common-use-cases)

### Configuration
- [CONFIGURATION_GUIDE.md § Configuration File Structure](CONFIGURATION_GUIDE.md#configuration-file-structure)
- [CONFIGURATION_GUIDE.md § Parameter Reference](CONFIGURATION_GUIDE.md#parameter-reference)
- [CONFIGURATION_GUIDE.md § Configuration Examples](CONFIGURATION_GUIDE.md#configuration-examples)

### Understanding Results
- [README.md § Understanding the Output](README.md#understanding-the-output)
- [USAGE_GUIDE.md § Interpreting Results](USAGE_GUIDE.md#interpreting-results)

### Advanced Usage
- [README.md § Advanced Usage](README.md#advanced-usage)
- [USAGE_GUIDE.md § Advanced Scenarios](USAGE_GUIDE.md#advanced-scenarios)
- [CONFIGURATION_GUIDE.md § Advanced Configuration](CONFIGURATION_GUIDE.md#advanced-configuration)

### Troubleshooting
- [README.md § Troubleshooting](README.md#troubleshooting)
- [USAGE_GUIDE.md § Troubleshooting Guide](USAGE_GUIDE.md#troubleshooting-guide)
- [CONFIGURATION_GUIDE.md § Troubleshooting Configuration](CONFIGURATION_GUIDE.md#troubleshooting-configuration)

### System Architecture
- [PROJECT_ANALYSIS_REPORT.md § Architecture](PROJECT_ANALYSIS_REPORT.md)
- [CODE_REVIEW.md § Code Structure](CODE_REVIEW.md)

### Development & Contributing
- [REFACTORING_PLAN.md](REFACTORING_PLAN.md)
- [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)
- [CODE_REVIEW.md](CODE_REVIEW.md)

---

## 🔍 Finding What You Need

### Common Questions

**Q: How do I install the software?**
→ [README.md § Installation](README.md#installation)

**Q: How do I run my first experiment?**
→ [README.md § Quick Start](README.md#quick-start)
→ [USAGE_GUIDE.md § Workflow 1](USAGE_GUIDE.md#workflow-1-first-time-user---running-all-shifts)

**Q: What do the results mean?**
→ [README.md § Understanding the Output](README.md#understanding-the-output)
→ [USAGE_GUIDE.md § Interpreting Results](USAGE_GUIDE.md#interpreting-results)

**Q: How do I configure the system?**
→ [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)

**Q: How do I run only specific shifts?**
→ [README.md § Step 4](README.md#step-4-run-specific-shifts-only)
→ [USAGE_GUIDE.md § Workflow 2](USAGE_GUIDE.md#workflow-2-testing-specific-hypothesis)

**Q: How do I make results reproducible?**
→ [README.md § Step 9](README.md#step-9-set-custom-random-seeds)
→ [USAGE_GUIDE.md § Use Case 1](USAGE_GUIDE.md#use-case-1-reproducibility-for-publication)
→ [CONFIGURATION_GUIDE.md § Reproducibility Parameters](CONFIGURATION_GUIDE.md#reproducibility-parameters)

**Q: Why is processing so slow?**
→ [README.md § Step 5](README.md#step-5-adjust-performance-parameters)
→ [USAGE_GUIDE.md § Workflow 3](USAGE_GUIDE.md#workflow-3-gpu-vs-cpu-comparison)
→ [CONFIGURATION_GUIDE.md § Performance Configuration](CONFIGURATION_GUIDE.md#performance-configuration)

**Q: What if I get an error?**
→ [README.md § Troubleshooting](README.md#troubleshooting)
→ [USAGE_GUIDE.md § Troubleshooting Guide](USAGE_GUIDE.md#troubleshooting-guide)

**Q: How do I customize shifts?**
→ [README.md § Custom Shift Implementation](README.md#custom-shift-implementation)
→ [USAGE_GUIDE.md § Scenario 1](USAGE_GUIDE.md#scenario-1-custom-shift-implementation)

**Q: How does the system work internally?**
→ [PROJECT_ANALYSIS_REPORT.md](PROJECT_ANALYSIS_REPORT.md)

**Q: What changes were made recently?**
→ [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)
→ [CODE_REVIEW.md](CODE_REVIEW.md)

---

## 📝 Documentation Standards

All documentation in this project follows these standards:

### Structure
- ✅ Clear table of contents
- ✅ Progressive disclosure (simple → complex)
- ✅ Code examples for all features
- ✅ Cross-references between documents

### Style
- ✅ Active voice
- ✅ Present tense
- ✅ Step-by-step instructions
- ✅ Visual indicators (✅ ❌ → etc.)

### Code Examples
- ✅ Runnable examples
- ✅ Expected output shown
- ✅ Error cases documented
- ✅ Platform-specific notes

### Maintenance
- ✅ Updated with each release
- ✅ Version-controlled
- ✅ Reviewed for accuracy

---

## 🔄 Documentation Versions

**Current Version:** 1.0.0 (Publication-ready)

**Last Updated:** December 2025

**Changelog:**
- v1.0.0: Complete documentation rewrite with step-by-step guides
- v0.9.0: Added configuration and usage guides
- v0.8.0: Initial documentation

---

## 💡 Contributing to Documentation

Found an error or want to improve the docs?

1. **Report issues:** Open a GitHub issue with the [documentation] tag
2. **Suggest improvements:** Create a pull request
3. **Ask questions:** If something is unclear, ask! We'll improve the docs

**Documentation Guidelines:**
- Use clear, simple language
- Include working code examples
- Test all commands before documenting
- Cross-reference related sections

---

## 📧 Get Help

Can't find what you need?

1. **Search this index** for your topic
2. **Check the troubleshooting sections**
3. **Open a GitHub issue** with:
   - What you're trying to do
   - What you've tried
   - Error messages (if any)
   - Your configuration

---

## 📚 External Resources

- [Original Clinical Behavioral Testing Paper](https://github.com/bvanaken/clinical-behavioral-testing)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [BioBERT Paper](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)

---

## Summary

**Essential Reading (30 min):**
1. README.md
2. USAGE_GUIDE.md (Basic Workflows)
3. CONFIGURATION_GUIDE.md (Quick reference)

**Complete Reading (1.5 hours):**
- All files in order listed above

**Quick Reference:**
- Use the "Documentation by Topic" section above
- Use "Common Questions" for FAQ

---

**Last Updated:** December 2025
**Version:** 1.0.0
**Status:** ✅ Complete and up-to-date

Built with ❤️ for the clinical NLP research community
