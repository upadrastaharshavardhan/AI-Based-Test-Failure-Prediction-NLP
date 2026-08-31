# Research Package - Project 5
## AI-Based Test Failure Prediction

Complete research paper, documentation, metrics, and full codebase.

**Key metrics:** Accuracy 93.40% | Macro F1 0.921 | Fail F1 0.887 | ROC-AUC 0.956

## Contents
- paper/ (PDF + MD)
- docs/
- results/
- codebase/ (full Project 5 source)

## Reproduce
```bash
cd codebase
pip install -r requirements.txt
python scripts/generate_data.py --n-samples 5000 --seed 42
python scripts/train.py
```
