from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import torch

from .engine import seed_everything
from .general_data import InstructionExample, render_instruction_prompt
from .model import CleoTransformer
from .tokenizer import ByteBPETokenizer


@dataclass(frozen=True)
class CapabilityGeneration:
    category: str
    prompt: str
    expected: str
    response: str
    passed: bool


def _held_out(key: str) -> bool:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 5 == 0


def build_capability_examples() -> tuple[list[InstructionExample], list[InstructionExample]]:
    train: list[InstructionExample] = []
    evaluation: list[InstructionExample] = []

    def add(category: str, key: str, prompt: str, response: str) -> None:
        example = InstructionExample(prompt, "", response, category)
        (evaluation if _held_out(f"{category}:{key}") else train).append(example)

    for left in range(21):
        for right in range(21):
            add(
                "addition",
                f"{left}+{right}",
                f"What is {left} plus {right}? Answer with only the number.",
                str(left + right),
            )
    for left in range(31):
        for right in range(left + 1):
            add(
                "subtraction",
                f"{left}-{right}",
                f"What is {left} minus {right}? Answer with only the number.",
                str(left - right),
            )
    for left in range(13):
        for right in range(13):
            add(
                "multiplication",
                f"{left}*{right}",
                f"What is {left} times {right}? Answer with only the number.",
                str(left * right),
            )
    for left in range(0, 100, 3):
        for right in range(1, 100, 7):
            if left == right:
                continue
            add(
                "comparison",
                f"{left}:{right}",
                f"Which number is larger: {left} or {right}? Answer with only the number.",
                str(max(left, right)),
            )

    positive = (
        "The service was excellent",
        "I loved the thoughtful result",
        "The meal was delicious",
        "Everything worked perfectly",
        "The support team was helpful",
        "This was a wonderful experience",
        "The update made the app faster",
        "I am happy with the outcome",
        "The instructions were clear",
        "The performance was impressive",
    )
    negative = (
        "The service was terrible",
        "I disliked the confusing result",
        "The meal was awful",
        "Nothing worked correctly",
        "The support team was unhelpful",
        "This was a frustrating experience",
        "The update made the app slower",
        "I am unhappy with the outcome",
        "The instructions were unclear",
        "The performance was disappointing",
    )
    for label, rows in (("positive", positive), ("negative", negative)):
        for index, sentence in enumerate(rows):
            add(
                "sentiment",
                f"{label}:{index}",
                f"Classify the sentiment as positive or negative: {sentence}.",
                label,
            )

    words = (
        "apple",
        "bridge",
        "camera",
        "delta",
        "energy",
        "forest",
        "garden",
        "harbor",
        "island",
        "jungle",
        "kitten",
        "lemon",
        "market",
        "notebook",
        "ocean",
        "planet",
        "quiet",
        "river",
        "silver",
        "tunnel",
        "umbrella",
        "violet",
        "window",
        "yellow",
        "zebra",
    )
    for word in words:
        add(
            "uppercase",
            word,
            f"Return this word in uppercase and nothing else: {word}",
            word.upper(),
        )

    names = ("Ava", "Ben", "Cara", "Drew", "Eli", "Faye", "Gus", "Hope")
    for index in range(120):
        name = names[index % len(names)]
        code = 1000 + index * 7
        context = (
            f"A routine note says the owner is {name}. "
            f"The verification code is {code}. Keep the number available."
        )
        add(
            "extraction",
            str(index),
            "Extract the verification code from the context. Answer with only the code.",
            str(code),
        )
        target = evaluation if _held_out(f"extraction:{index}") else train
        target[-1] = InstructionExample(
            target[-1].instruction,
            context,
            target[-1].response,
            target[-1].category,
        )

    key = lambda example: hashlib.sha256(
        json.dumps(asdict(example), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return sorted(train, key=key), sorted(evaluation, key=key)


def normalize_capability_response(value: str, *, case_sensitive: bool = False) -> str:
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    normalized = first_line.strip().rstrip(".!")
    return normalized if case_sensitive else normalized.casefold()


def select_capability_examples(
    examples: Iterable[InstructionExample], limit: int
) -> list[InstructionExample]:
    """Select a deterministic, category-balanced benchmark slice."""
    if limit < 1:
        raise ValueError("capability evaluation limit must be positive")
    by_category: dict[str, list[InstructionExample]] = {}
    for example in examples:
        by_category.setdefault(example.category, []).append(example)
    selected: list[InstructionExample] = []
    categories = sorted(by_category)
    index = 0
    while len(selected) < limit and categories:
        remaining: list[str] = []
        for category in categories:
            rows = by_category[category]
            if index < len(rows) and len(selected) < limit:
                selected.append(rows[index])
            if index + 1 < len(rows):
                remaining.append(category)
        categories = remaining
        index += 1
    return selected


@torch.no_grad()
def evaluate_capability_generation(
    model: CleoTransformer,
    tokenizer: ByteBPETokenizer,
    examples: Iterable[InstructionExample],
    device: torch.device,
    *,
    limit: int = 128,
    seed: int = 1337,
    max_new_tokens: int = 24,
) -> list[CapabilityGeneration]:
    rows = select_capability_examples(examples, limit)
    if not rows:
        raise ValueError("capability evaluation requires at least one example")
    model.eval()
    results: list[CapabilityGeneration] = []
    for index, example in enumerate(rows):
        seed_everything(seed + index)
        rendered = render_instruction_prompt(example.instruction, example.context)
        prompt_ids = tokenizer.encode(rendered, bos=True)
        if len(prompt_ids) > model.config.block_size:
            prompt_ids = prompt_ids[-model.config.block_size :]
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            inputs,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            min_new_tokens=0,
            use_cache=True,
        )
        full_text = tokenizer.decode(generated[0].tolist())
        response = full_text[len(rendered) :].strip()
        case_sensitive = example.category == "uppercase"
        passed = normalize_capability_response(
            response, case_sensitive=case_sensitive
        ) == normalize_capability_response(
            example.response, case_sensitive=case_sensitive
        )
        results.append(
            CapabilityGeneration(
                category=example.category,
                prompt=example.instruction,
                expected=example.response,
                response=response,
                passed=passed,
            )
        )
    return results


def capability_accuracy(results: Iterable[CapabilityGeneration]) -> float:
    rows = list(results)
    if not rows:
        raise ValueError("capability accuracy requires at least one result")
    return sum(row.passed for row in rows) / len(rows)
