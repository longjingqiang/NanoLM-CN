# Checkpoints Directory

Model weight files (`.pt`) and tokenizer binaries are excluded from the
repository via `.gitignore` (they total over 2 GB and exceed GitHub's
recommended file size limits).

## What's included in the repo

The following small files are committed to document the training run:

- `train_config.json` — full training hyperparameters
- `training_log.json` — train/val loss curves over all steps

## How to obtain the trained model

Re-train from scratch (see project root `README.md` for instructions),
or — if a release is published — download the checkpoint from the
GitHub Releases page and place it here:

```
checkpoints/
├── best.pt              # main inference checkpoint (~145 MB)
├── tokenizer_0.5M/      # BPE tokenizer (~300 KB)
│   ├── tokenizer.json
│   └── ...
├── train_config.json    # (in repo)
└── training_log.json    # (in repo)
```

## Training summary (small model)

- Parameters: ~37M (8 layers, 512 dim, vocab 8000)
- Steps: 20000
- Best validation loss: 1.446
- Training time: ~6 hours on RTX 5060 Laptop (8 GB)
- Dataset: BELLE 0.5M Chinese instructions

See `docs/inference_test_report.md` for qualitative evaluation results.
