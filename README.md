# Titanic – Data Preprocessing & ML

A clean, portfolio-ready repo for Titanic classification experiments.
Two Google Colab notebooks are linked via badges, plus a local scikit-learn pipeline.

## Notebooks (Colab)
- Baseline: https://colab.research.google.com/drive/1GBJfgZN4xxxZX7LPAjmtD1Sy_YR2L00O?usp=sharing
- Improved: https://colab.research.google.com/drive/1b5S5wHqXlsQuX2oxqABmvkmSIrtJaygk?usp=sharing

## Local quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put Kaggle Titanic CSV as data/titanic.csv, then:
bash scripts/run_all.sh
```

## Repo layout
- notebooks/ — baseline & improved Colab notebooks (badges)
- src/ — train.py to run 9 models locally
- scripts/ — run_all.sh
- data/ — README with data instructions
- docs/ — assignment brief & report PDFs
- requirements.txt — Python dependencies

Brief and report PDFs reflect the assignment prompt and your write-up.
