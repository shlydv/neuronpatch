"""Load exactly the audited upstream metric functions without unrelated imports."""

import ast
import hashlib

import numpy as np
import torch

SOURCE_SHA256 = "d57aba0656ee38a12dc9dcf3dc071ff77b1dc2c33a4c9de495241d5aabe173e2"


def load_metric(root):
    path = root / "knowledge-editing/easyeditor/evaluate/evaluate_utils.py"
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Upstream evaluator differs from audited source")
    names = {"test_prediction_acc", "slice_list_leftpad", "slice_list_rightpad"}
    tree = ast.parse(source, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["test_prediction_acc"]


def first_answer(value):
    """Match the published run.py's first nested alias policy, not best alias."""
    while isinstance(value, list):
        if not value:
            raise ValueError("Empty answer list")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing answer text")
    return value


def score_probes(metric, model, tokenizer, hparams, probes):
    results = []
    for probe in probes:
        answer = first_answer(probe["gold"])
        args = (model, tokenizer, hparams, probe["prompt"], answer, hparams.device)
        results.append(
            {
                "target": answer,
                "teacher_forced_token_accuracy": metric(*args),
                "teacher_forced_predictions": metric(*args, locality=True),
            }
        )
    return results


def locality_agreement(before, after):
    if len(before) != len(after):
        raise ValueError("Prediction row counts differ")
    values = []
    for old, new in zip(before, after, strict=True):
        if len(old) != len(new) or not old:
            raise ValueError("Prediction token counts differ or are empty")
        values.append(float(np.mean(np.equal(old, new))))
    return values
