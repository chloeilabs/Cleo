from __future__ import annotations

from pathlib import Path

import torch

from cleo1.identity import (
    CANONICAL_IDENTITY_RESPONSE,
    COMPANY_NAME,
    IDENTITY_EVAL_EXAMPLES,
    MODEL_ID,
    MODEL_NAME,
    identity_facts_present,
    identity_leakage_present,
    identity_response_matches,
    identity_response_for_prompt,
    is_identity_question,
    model_identity_metadata,
)
from cleo1.identity_tuning import IGNORE_INDEX, encode_identity_example, identity_batch
from cleo1.tokenizer import ByteBPETokenizer


TOKENIZER_PATH = Path("data/processed/tokenizer.json")


def test_canonical_identity_is_exact_and_self_describing():
    metadata = model_identity_metadata()
    assert metadata["company_name"] == COMPANY_NAME == "Cleo AI"
    assert metadata["model_name"] == MODEL_NAME == "Cleo 1"
    assert metadata["model_id"] == MODEL_ID == "cleo-1"
    assert metadata["canonical_response"] == CANONICAL_IDENTITY_RESPONSE
    assert identity_facts_present(CANONICAL_IDENTITY_RESPONSE)
    assert identity_response_matches(CANONICAL_IDENTITY_RESPONSE)
    assert not identity_response_matches(
        "I am Cleo 1 9model ID: cleo-1), a model developed by Cleo AI."
    )
    assert identity_leakage_present("The bird said, I am Cleo, and flew away.")
    assert identity_leakage_present("My model ID is cleo-1.")
    assert identity_leakage_present("The dog was trained to find a shiny key.")
    assert not identity_leakage_present("The dog learned to find a shiny key.")


def test_identity_question_detection_is_narrow():
    assert is_identity_question("What is your model name and who trained you?")
    assert is_identity_question("Who developed this model?")
    assert is_identity_question(
        "Which organization trained this system, and what are its exact name and ID?"
    )
    assert is_identity_question(
        "Who is responsible for training the model currently answering?"
    )
    assert identity_response_for_prompt("Identify yourself.") == CANONICAL_IDENTITY_RESPONSE
    assert not is_identity_question("Once upon a time, Cleo found a shiny key")
    assert identity_response_for_prompt("Write a story about a robot") is None


def test_identity_examples_mask_prompt_loss_and_fit_context():
    tokenizer = ByteBPETokenizer.load(TOKENIZER_PATH)
    example = IDENTITY_EVAL_EXAMPLES[0]
    encoded = encode_identity_example(tokenizer, example, block_size=256)
    assert len(encoded.inputs) == len(encoded.targets)
    assert len(encoded.inputs) <= 256
    assert encoded.targets[0] == IGNORE_INDEX
    assert any(target != IGNORE_INDEX for target in encoded.targets)
    assert example.prompt in tokenizer.decode(encoded.inputs)

    inputs, targets = identity_batch(
        tokenizer,
        IDENTITY_EVAL_EXAMPLES[:2],
        block_size=256,
        device=torch.device("cpu"),
    )
    assert inputs.shape == targets.shape
    assert inputs.size(0) == 2
    assert bool((targets == IGNORE_INDEX).any())
