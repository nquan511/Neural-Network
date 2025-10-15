# Copilot Instructions for Neural Network Codebase

## Overview
This workspace contains a collection of neural network experiments, educational notebooks, and prototype models. The main focus is on language modeling (GPT, bigram, transformer), time series modeling, and practical PyTorch implementations. Most workflows are notebook-driven, with supporting Python scripts for reusable components.

## Key Components
- **Notebooks**: Each notebook (e.g., `gpt.ipynb`, `makemore_*.ipynb`, `micrograd_*.ipynb`, `financial_attention/test.ipynb`) demonstrates a self-contained experiment or tutorial. Notebooks are the primary entry point for exploration and development.
- **Scripts**: Files like `bigram.py`, `v2.py`, and `financial_attention/financial_gpt.py` contain reusable model code or utilities referenced in notebooks.
- **Data**: Text files (`input.txt`, `names.txt`) and CSVs (`financial_attention/1h_data_20220101_20250601.csv`) are used for training and evaluation. Paths are often hardcoded in notebooks/scripts.

## Developer Workflows
- **Experimentation**: Start with a notebook. Run cells sequentially; most notebooks are designed to be executed top-to-bottom. Modify parameters or code directly in cells for rapid iteration.
- **Training**: Model training is typically performed in notebooks. For time series, use `financial_attention/test_enc_dec.ipynb` and related scripts. For language models, use `gpt.ipynb` or `makemore_*.ipynb`.
- **Testing**: There is no formal test suite. Validation is performed by inspecting outputs in notebooks or running scripts interactively.
- **Debugging**: Use print statements and cell outputs. PyTorch models are often defined inline for transparency.
- **Builds**: No build system; dependencies are managed manually. Install packages (e.g., `torch`, `tiktoken`) as needed using pip.

## Project-Specific Patterns
- **Manual Data Loading**: Data files are loaded with explicit paths (e.g., `with open('input.txt')`). Update paths if moving files or running in a different directory.
- **Minimal Abstraction**: Most code is written for clarity and educational value, not for reuse. Expect repetition and direct implementation of algorithms.
- **PyTorch Usage**: Models are defined as subclasses of `nn.Module`. Training loops are written explicitly in cells/scripts.
- **Tokenizer Experiments**: Notebooks like `gpt.ipynb` show both custom and GPT2 tokenization (using `tiktoken`).
- **Time Series Models**: The `financial_attention/` directory contains transformer-based models for financial data. See `transformer_time_series.py` and related notebooks.

## Integration Points
- **External Libraries**: Key dependencies include `torch`, `tiktoken`, and standard Python libraries. Install manually as needed.
- **Model Checkpoints**: Pretrained weights (e.g., `gpt_ts_best.pth`) are loaded in scripts/notebooks for evaluation or fine-tuning.

## Examples
- To run a GPT-style experiment: open `gpt.ipynb`, ensure `input.txt` is present, and run all cells.
- For financial time series modeling: use `financial_attention/test_enc_dec.ipynb` and ensure the CSV data is available.

## Conventions
- Notebooks are the canonical source of truth; scripts support but do not replace them.
- Data paths are relative to the notebook/script location.
- No formal testing, linting, or CI/CD.

---
For questions or unclear patterns, review the relevant notebook or script directly. Update this file as new conventions emerge.