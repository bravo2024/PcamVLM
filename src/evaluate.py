"""Evaluation metrics for PcamVLM.

Reports accuracy + sensitivity + specificity + precision + F1 for binary
tasks (PCam), and macro-F1 + per-class accuracy for the 9-class NCT-CRC
task. The Streamlit dashboard renders the resulting confusion matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class EvalReport:
    task: str
    n: int
    accuracy: float
    per_class: Dict[int, Dict[str, float]] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "n": self.n,
            "accuracy": self.accuracy,
            "per_class": {str(k): v for k, v in self.per_class.items()},
            "confusion": self.confusion,
            "class_names": self.class_names,
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate_binary(trues: Sequence[int], preds: Sequence[int]) -> EvalReport:
    """Binary classification metrics (PCam)."""
    n = len(trues)
    if n == 0:
        return EvalReport(task="pcam", n=0, accuracy=0.0,
                          per_class={}, confusion=[[0, 0], [0, 0]],
                          class_names=["no", "yes"])
    tp = sum(1 for t, p in zip(trues, preds) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(trues, preds) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(trues, preds) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(trues, preds) if t == 1 and p == 0)
    accuracy = (tp + tn) / n
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    prec = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * prec * sens, prec + sens)
    return EvalReport(
        task="pcam",
        n=n,
        accuracy=accuracy,
        per_class={
            0: {"precision": spec, "recall": spec, "f1": spec,
                "support": tn + fp, "name": "no"},
            1: {"precision": prec, "recall": sens, "f1": f1,
                "support": tp + fn, "name": "yes"},
        },
        confusion=[[tn, fp], [fn, tp]],
        class_names=["no", "yes"],
    )


def evaluate_multiclass(trues: Sequence[int], preds: Sequence[int],
                        class_names: Sequence[str]) -> EvalReport:
    """Macro-F1 + per-class accuracy + confusion matrix."""
    n = len(trues)
    k = len(class_names)
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(trues, preds):
        cm[int(t)][int(p)] += 1
    if n == 0:
        return EvalReport(task="nct_crc", n=0, accuracy=0.0,
                          per_class={i: {"name": name} for i, name in enumerate(class_names)},
                          confusion=cm, class_names=list(class_names))
    correct = sum(cm[i][i] for i in range(k))
    accuracy = correct / n
    per_class: Dict[int, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        support = sum(cm[i])
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(k) if r != i)
        fn = support - tp
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        per_class[i] = {"name": name, "support": support,
                        "precision": prec, "recall": rec, "f1": f1}
    return EvalReport(
        task="nct_crc", n=n, accuracy=accuracy,
        per_class=per_class, confusion=cm, class_names=list(class_names),
    )


def evaluate_task(task: str, trues: Sequence[int], preds: Sequence[int],
                  class_names: Sequence[str] | None = None) -> EvalReport:
    if task == "pcam":
        return evaluate_binary(trues, preds)
    if task == "nct_crc":
        cn = list(class_names) if class_names else [
            "ADI", "BACK", "DEB", "LYM", "MUC",
            "MUS", "NORM", "STR", "TUM",
        ]
        return evaluate_multiclass(trues, preds, cn)
    raise ValueError(f"unknown task: {task!r}")


__all__ = [
    "EvalReport",
    "evaluate_binary",
    "evaluate_multiclass",
    "evaluate_task",
]