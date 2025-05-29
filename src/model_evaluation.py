# src/model_evaluation.py

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.utils import setup_logging, FIGURES_PATH

logger = setup_logging(__name__)

MODELS_PATH = FIGURES_PATH / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

def save_confusion_matrix(name: str, y_true, y_pred) -> None:
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Stay","Churn"])
    ax.set_yticklabels(["Stay","Churn"])
    for i in (0,1):
        for j in (0,1):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="white")
    out = MODELS_PATH / f"{name.lower()}_cm.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved model-eval plot → {out}")

def save_roc_comparison(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(8,6))
    for name, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    out = MODELS_PATH / "roc_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved model-eval plot → {out}")
