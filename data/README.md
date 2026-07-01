# Data Directory

This directory is intentionally empty in the repository — raw and processed data
files are excluded by `.gitignore` because they are too large to host on GitHub.

## How to populate this directory

### 1. Download the BELLE dataset

Download `Belle_open_source_0.5M.json` (~273 MB) from the BELLE project:

- BELLE GitHub: https://github.com/LianjiaTech/BELLE
- HuggingFace: https://huggingface.co/datasets/BelleGroup/train_0.5M_CN

Place the file at:

```
data/raw/Belle_open_source_0.5M.json
```

### 2. Run preprocessing

```bash
python scripts/prepare_instruction_data.py
```

This will produce:

```
data/processed_0.5M/train.bin     # ~185 MB
data/processed_0.5M/val.bin       # ~1.9 MB
```

### Expected layout

```
data/
├── raw/
│   └── Belle_open_source_0.5M.json
└── processed_0.5M/
    ├── train.bin
    └── val.bin
```

## License note

The BELLE dataset was generated using OpenAI's ChatGPT API and is restricted
to research and non-commercial use. See the project root `NOTICE` file for
full attribution.
