# AI-Based Test Failure Prediction

**Project 5** – Predict whether a test case will pass or fail **before execution** using NLP on test metadata, recent changes, and historical signals.

## What it does

Given a test case (name, module, recent commit message, changed files, history summary), the system predicts:

1. **Outcome**: Pass or Fail
2. **Confidence** score
3. **Similar historical failing tests** (optional retrieval)

## Key Features

- Synthetic test-case generator with realistic pass/fail patterns
- Sentence embeddings of test name + change context
- Binary classifier (Logistic Regression / Random Forest)
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Gradio demo
- Colab-ready modular structure

## Quick Start

```bash
!pip install -r requirements.txt
!python scripts/generate_data.py --n-samples 5000
!python scripts/train.py
!python -m src.api.gradio_app
```

## Example

```python
from src.pipeline.predictor import TestFailurePredictor
predictor = TestFailurePredictor.load("artifacts")
result = predictor.predict(
    test_name="test_payment_refund_flow",
    module="payment-service",
    commit_message="Refactor refund API timeout handling",
    changed_files="PaymentService.java, RefundController.java"
)
print(result)  # {"prediction": "fail", "confidence": 0.81, ...}
```

## License

MIT
