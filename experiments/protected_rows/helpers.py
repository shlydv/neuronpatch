"""Prompt-aligned single-example training and explicit generated-answer scoring."""

import json
from pathlib import Path

import torch

from experiments.real_qa.data import prompt, score
from experiments.real_qa.response_loss import response_ce, response_logits


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class Examples:
    def __init__(self, tokenizer, device, *, generation_tokens=32):
        self.tokenizer = tokenizer
        self.device = device
        self.generation_tokens = generation_tokens
        self.cache = {}

    def prefix(self, row):
        if row["id"] not in self.cache:
            text = (
                prompt(self.tokenizer, row)
                if "context" in row
                else self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            self.cache[row["id"]] = self.tokenizer.encode(text, add_special_tokens=False)
        return list(self.cache[row["id"]])

    def inputs(self, row, *, training=False):
        prefix = self.prefix(row)
        target = (
            (
                row["_target_ids"]
                if "_target_ids" in row
                else self.tokenizer.encode(row["answer"], add_special_tokens=False)
                + [self.tokenizer.eos_token_id]
            )
            if training
            else []
        )
        ids = torch.tensor([prefix + target], device=self.device, dtype=torch.long)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if training:
            inputs["labels"] = torch.tensor(
                [[-100] * len(prefix) + target], device=self.device, dtype=torch.long
            )
        return inputs

    def loss(self, model, row):
        logits, labels = response_logits(model, self.inputs(row, training=True))
        return response_ce(logits, labels)

    @torch.no_grad()
    def generate(self, model, row):
        inputs = self.inputs(row)
        output = model.generate(
            **inputs,
            max_new_tokens=self.generation_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        tokens = output[0, inputs["input_ids"].shape[1] :].cpu().tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if "context" in row:
            values = score(text, row["answers"])
        else:
            predicted = text.strip().rstrip(".!?")
            expected = row["answers"]
            # Uppercase is a case-sensitive task; SQuAD normalization would erase it.
            hit = (
                predicted in expected
                if row["kind"] == "text_operation"
                else predicted.casefold() in {a.casefold() for a in expected}
            )
            values = {
                "em": float(hit),
                "f1": float(hit),
                "abstained": predicted.upper() == "UNKNOWN",
            }
        return {
            "id": row["id"],
            "kind": row["kind"],
            "article": row.get("article"),
            "text": text,
            "answers": row["answers"],
            "generated_token_ids": tokens,
            **values,
        }

    def evaluate(self, model, rows, path, *, budget=lambda: None):
        values = []
        for index, row in enumerate(rows):
            budget()
            values.append(self.generate(model, row))
            if (index + 1) % 32 == 0:
                print(f"STAGE evaluate={Path(path).stem} examples={index + 1}", flush=True)
        write_json(path, values)
        return values
