"""Matched hard/soft teacher supervision on a reviewed, frozen development bank."""

import torch

from neuronpatch.patch import fingerprint
from neuronpatch.supervision import ResponseExample, chat_response

from .checkpointing import enable_decoder_recomputation
from .preference_experiment import run_comparison, write
from .preference_run import generate, main
from .preference_training import PreparedPair, digest


def exact_continuation(tokenizer, messages, tokens, *, max_length, device):
    """Keep the archived generated tokens rather than decode/retokenize them."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    if not prompt or not tokens or tokens[-1] != tokenizer.eos_token_id:
        raise ValueError("Nonempty prompt and EOS-terminated teacher required")
    if tokenizer.eos_token_id in tokens[:-1] or len(prompt) + len(tokens) > max_length:
        raise ValueError("Internal EOS or context overflow in teacher trajectory")
    ids = torch.tensor([prompt + tokens], dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, len(prompt) :] = True
    return ResponseExample(ids, torch.ones_like(mask), mask, len(prompt))


def prepare_samples(tokenizer, bank, reviewed, *, max_length, device):
    """Use identical response continuations for clean and attacked contexts."""
    tasks = {r["document_id"]: r for r in bank["tasks"]}
    accepted = {r["task"]["document_id"]: r for r in reviewed}
    if len(accepted) != len(reviewed) or not accepted or len(tasks) != len(bank["tasks"]):
        raise ValueError("Unique, nonempty task and teacher banks required")
    clean, indices = [], {}
    for did, row in accepted.items():
        if tasks.get(did) != row["task"]:
            raise ValueError("Reviewed task differs from current bank")
        teacher = row["teacher"]
        if (
            not teacher["ended"]
            or tokenizer.decode(teacher["tokens"], skip_special_tokens=True) != teacher["text"]
        ):
            raise ValueError("Teacher decoding or termination mismatch")
        sample = exact_continuation(
            tokenizer,
            row["task"]["messages"],
            teacher["tokens"],
            max_length=max_length,
            device=device,
        )
        if sample.input_ids[0, : sample.prompt_length].tolist() != teacher["prompt_ids"]:
            raise ValueError("Teacher prompt differs under pinned tokenizer")
        indices[did] = len(clean)
        clean.append(sample)
    pairs, mapping = [], {}
    counts = {(did, trusted): 0 for did in accepted for trusted in (False, True)}
    for row in bank["diverse_pairs"]:
        did = row["document_id"]
        if did not in accepted:
            continue
        if row["id"] in mapping or type(row["trusted"]) is not bool:
            raise ValueError("Duplicate pair or invalid trust annotation")
        teacher = accepted[did]["teacher"]
        target = exact_continuation(
            tokenizer,
            row["messages"],
            teacher["tokens"],
            max_length=max_length,
            device=device,
        )
        marker = chat_response(
            tokenizer,
            row["messages"],
            row["chosen"] if row["trusted"] else row["rejected"],
            max_length=max_length,
            device=device,
        )
        chosen, rejected = (marker, target) if row["trusted"] else (target, marker)
        identity = digest(
            {
                "id": row["id"],
                "chosen": chosen.input_ids.tolist(),
                "rejected": rejected.input_ids.tolist(),
                "prompt_length": chosen.prompt_length,
            }
        )
        pairs.append(PreparedPair(row["id"], identity, chosen, rejected))
        mapping[row["id"]] = None if row["trusted"] else indices[did]
        counts[(did, row["trusted"])] += 1
    if set(counts.values()) != {3}:
        raise ValueError("Require three trusted and three untrusted examples per document")
    return pairs, clean, mapping


def execute(model, tokenizer, data, spec, output, check_time):
    if fingerprint(model) != spec["base_fingerprint"]:
        raise ValueError("Loaded base fingerprint mismatch")
    reviewed = data["teachers"]
    if any(reviewed[key] != spec[key] for key in ("model", "revision")) or (
        reviewed["recorded_base_fingerprint"] != spec["base_fingerprint"]
    ):
        raise ValueError("Reviewed bank model identity mismatch")
    membership = [
        {t["document_id"] for t in data[role]["tasks"]}
        for role in ("train", "calibration", "confirmation")
    ]
    if any(membership[i] & membership[j] for i in range(3) for j in range(i)):
        raise ValueError("Document leakage across roles")
    pairs, clean, mapping = prepare_samples(
        tokenizer,
        data["train"],
        reviewed["teachers"]["train"],
        max_length=spec["max_sequence_tokens"],
        device=model.device,
    )
    if len(pairs) != spec["training"]["steps"] or len(clean) != spec["training_documents"]:
        raise ValueError("Training membership differs from frozen count")
    by_id = {r["id"]: r for r in data["confirmation"]["cases"]}
    cases = [by_id[i] for i in spec["diagnostic_case_ids"]]
    if len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Duplicate evaluation cases")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", spec)
    write(output / "teacher-preparation.json", reviewed)
    enable_decoder_recomputation(model)

    def evaluate(fitted, directory):
        return generate(
            fitted,
            tokenizer,
            cases,
            spec["generation"],
            directory / "confirmation.json",
            check_time,
        )

    base_dir = output / "base"
    base_dir.mkdir()
    write(base_dir / "result.json", evaluate(model, base_dir))
    names = []
    for i in range(len(model.model.layers)):
        names.extend(f"model.layers.{i}.self_attn.{p}_proj" for p in ("q", "k", "v", "o"))
        names.extend(f"model.layers.{i}.mlp.{p}_proj" for p in ("gate", "up", "down"))
    model, results = run_comparison(
        model,
        {"narrow": pairs, "diverse": pairs},
        clean,
        names=names,
        rank=spec["rank"],
        protocol=spec["training"],
        output=output / "comparison",
        evaluate=evaluate,
        check_time=check_time,
        arms=[("diverse", objective, "combined") for objective in ("sft", "cross_context")],
        preservation_objective="kl",
        max_cache_bytes=spec["max_retention_cache_bytes"],
        task_teacher_indices=mapping,
    )
    write(
        output / "complete.json",
        {
            "fits": len(results),
            "base_restored": fingerprint(model) == spec["base_fingerprint"],
            "scope": "Exposed development cases; not fresh confirmation or sparse superiority",
        },
    )


if __name__ == "__main__":
    main(execute_fn=execute)
