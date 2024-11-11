# Clinical Valence Testing with Attention Analysis

This project extends the work of [Clinical Behavioral Testing](https://github.com/bvanaken/clinical-behavioral-testing) by van Aken et al. to examine how valence words affect clinical predictions and analyze attention patterns in clinical NLP models. The original work focused on behavioral testing of clinical NLP models, while this extension adds attention analysis capabilities to understand how model predictions are influenced by specific language patterns.

## Citation
```bibtex
@inproceedings{vanAken2021,
   author    = {Betty van Aken and
                Sebastian Herrmann and
                Alexander Löser},
   title     = {What Do You See in this Patient? Behavioral Testing of Clinical NLP Models},
   booktitle = {Bridging the Gap: From Machine Learning Research to Clinical Practice,
                Research2Clinics Workshop @ NeurIPS 2021},
   year      = {2021}
}
```

## Features

- **Enhanced Behavioral Testing**: Building on the original framework with additional valence testing capabilities
- **Attention Analysis**: Examination of model attention patterns to understand prediction influences
- **Multiple Shift Types**: 
  - Neutralization
  - Pejorative language analysis
  - Laudatory language analysis
  - Neutral value testing
- **Comprehensive Output**: Generates detailed results including:
  - Diagnosis predictions
  - Attention weights
  - Clinical note analysis
  - Statistical summaries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gyasifred/clinical-valence-testing.git
cd clinical-valence-testing
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Entry point for running behavioral tests
- `prediction.py`: Contains predictor classes for model inference
- `utils.py`: Utility functions for data processing and analysis
- `valence_testing.py`: Core behavioral testing implementation
- `test_shifts/`: Directory containing different shift implementations:
  - `laudatory_shift.py`
  - `neutralize_shift.py`
  - `neutralVal_shift.py`
  - `pejorative_shift.py`

## Usage

### Basic Usage

Run behavioral tests with specific shifts:

```bash
python main.py \
  --test_set_path /path/to/test/data.csv \
  --model_path /path/to/model/checkpoint \
  --shift_keys neutralize,pejorative,laud,neutralval \
  --task diagnosis \
  --save_dir ./results
```

### Parameters

- `test_set_path`: Path to test dataset CSV
- `model_path`: Path to model checkpoint
- `shift_keys`: Comma-separated list of shifts to apply
- `task`: Prediction task (currently supports "diagnosis")
- `save_dir`: Directory to save results
- `gpu`: Whether to use GPU (default: False)
- `batch_size`: Batch size for predictions (default: 128)
- `head_num`: Attention head to analyze (default: 11)
- `layer_num`: Model layer to analyze (default: 11)
- `code_label`: Column name for codes in dataset (default: "short_codes")
- `checkpoint_interval`: Interval for saving checkpoints (default: 1000)

## Development Status

⚠️ **This project is currently under active development** ⚠️

Please note that this is a research project in development. Features, APIs, and documentation may change significantly. Use with caution in production environments.

## License

This project follows the licensing terms of the original Clinical Behavioral Testing project.