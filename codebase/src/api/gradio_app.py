from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gradio as gr
from src.pipeline.predictor import TestFailurePredictor
from src.utils.helpers import load_config

def build_demo(artifacts_dir="artifacts", config_path="config/config.yaml"):
    cfg = load_config(config_path)
    predictor = TestFailurePredictor.load(artifacts_dir, config_path)

    def predict_fn(test_name, module, commit_message, changed_files):
        r = predictor.predict(test_name=test_name, module=module,
                              commit_message=commit_message, changed_files=changed_files)
        lines = [
            f"### Prediction: **{r['prediction'].upper()}**",
            f"**Confidence:** {r['confidence']:.1%}",
            "",
            "### Similar historical tests",
        ]
        for i, s in enumerate(r.get("similar_tests", []), 1):
            lines.append(f"{i}. `{s['test_id']}` | {s['test_name']} | **{s['outcome']}** | sim={s['similarity']:.3f}")
        return "\n".join(lines)

    demo = gr.Interface(
        fn=predict_fn,
        inputs=[
            gr.Textbox(label="Test name", value="test_payment_refund_flow"),
            gr.Textbox(label="Module", value="payment-service"),
            gr.Textbox(label="Commit message", value="Refactor refund API timeout handling"),
            gr.Textbox(label="Changed files", value="PaymentService.java, RefundController.java"),
        ],
        outputs=gr.Markdown(),
        title=cfg.get("gradio", {}).get("title", "AI Test Failure Predictor"),
        description=cfg.get("gradio", {}).get("description", ""),
        examples=[
            ["test_payment_refund_flow", "payment-service", "Refactor refund API timeout handling", "PaymentService.java"],
            ["test_health_endpoint", "api-gateway", "Update README and docs", "README.md"],
            ["test_auth_token_refresh", "auth-service", "Fix JWT expiration edge case", "AuthService.java"],
        ],
        allow_flagging="never",
    )
    return demo

if __name__ == "__main__":
    build_demo().launch(share=False, server_name="0.0.0.0")
