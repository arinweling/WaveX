# WaveX (CartoonX / Wavelet Explanations)

This repo contains experiments and tooling around CartoonX-style wavelet explanations.

## Repo layout

```
.
├── docs/                        # Papers, prompts, notes
│   ├── CartoonX.pdf
│   └── wavelet_idea2_implementation_prompt.md
├── requirements.txt             # Dependencies
└── wavelet_explanation/         # Main codebase + configs
    ├── cartoonx.py
    ├── cartoonx_config.yaml
    ├── train.py / train_*.py
    ├── evaluate.py
    ├── configs/
    ├── models/
    ├── training/
    ├── tests/
    └── data/                    # Ignored (datasets)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick usage (example)

```bash
cd wavelet_explanation
python cartoonx.py --config cartoonx_config.yaml
```

## Notes
- Large datasets and generated outputs are git-ignored (`wavelet_explanation/outputs/`).
- Place new docs in `docs/`.
