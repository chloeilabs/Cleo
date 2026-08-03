from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import torch

from cleo1.capabilities import (
    build_capability_examples,
    normalize_capability_response,
    select_capability_examples,
)
from cleo1.checkpoint import atomic_torch_save
from cleo1.config import AppConfig, DataConfig, ModelConfig, TrainingConfig
from cleo1.general_data import (
    GeneralConfig,
    GeneralDataConfig,
    GeneralTrainingConfig,
    IGNORE_INDEX,
    InstructionExample,
    encode_instruction_example,
    load_general_config,
    split_instruction_examples,
)
import cleo1.general_training as general_training
from cleo1.general_training import expand_context_model
from cleo1.identity import model_identity_metadata
from cleo1.identity_tuning import IdentityGeneration
from cleo1.model import CleoTransformer
from cleo1.tokenizer import ByteBPETokenizer


def _tokenizer() -> ByteBPETokenizer:
    corpus = (
        "Instruction response context general knowledge arithmetic science "
        "classification explanation Cleo AI. " * 40
    ).encode()
    return ByteBPETokenizer.train(corpus, vocab_size=320)


def test_general_config_is_pinned_and_valid() -> None:
    config = load_general_config("configs/general_m4.toml")
    assert config.data.general_dataset == "Salesforce/wikitext"
    assert len(config.data.general_revision) == 40
    assert config.training.block_size == 512
    assert config.training.effective_batch_tokens % config.training.block_size == 0


def test_capability_curriculum_is_deterministic_and_held_out() -> None:
    first_train, first_evaluation = build_capability_examples()
    second_train, second_evaluation = build_capability_examples()
    assert (first_train, first_evaluation) == (second_train, second_evaluation)
    assert len(first_train) > 1_000
    assert len(first_evaluation) > 200
    train_rows = {(row.category, row.instruction, row.context) for row in first_train}
    evaluation_rows = {
        (row.category, row.instruction, row.context) for row in first_evaluation
    }
    assert train_rows.isdisjoint(evaluation_rows)
    assert {row.category for row in first_evaluation} >= {
        "addition",
        "subtraction",
        "multiplication",
        "comparison",
        "sentiment",
        "uppercase",
        "extraction",
    }
    assert normalize_capability_response("  Positive.\nExtra") == "positive"
    assert normalize_capability_response("APPLE", case_sensitive=True) == "APPLE"
    assert normalize_capability_response("apple", case_sensitive=True) != "APPLE"
    balanced = select_capability_examples(first_evaluation, 35)
    assert len({row.category for row in balanced}) == 7


def test_instruction_split_is_deterministic_and_disjoint() -> None:
    examples = [
        InstructionExample(
            instruction=f"Question {index}",
            context="",
            response=f"Answer {index}",
            category="qa" if index % 2 else "classification",
        )
        for index in range(40)
    ]
    first = split_instruction_examples(examples)
    second = split_instruction_examples(reversed(examples))
    assert first == second
    assert [len(split) for split in first] == [36, 2, 2]
    identities = [
        {example.instruction for example in split}
        for split in first
    ]
    assert identities[0].isdisjoint(identities[1])
    assert identities[0].isdisjoint(identities[2])
    assert identities[1].isdisjoint(identities[2])


def test_instruction_encoding_masks_prompt_and_truncates_losslessly() -> None:
    tokenizer = _tokenizer()
    example = InstructionExample(
        instruction="Explain the supplied context.",
        context="context " * 200,
        response="A concise response. " * 40,
        category="summarization",
    )
    encoded = encode_instruction_example(tokenizer, example, block_size=128)
    assert len(encoded.inputs) <= 128
    assert len(encoded.inputs) == len(encoded.targets)
    assert encoded.targets[0] == IGNORE_INDEX
    assert any(target != IGNORE_INDEX for target in encoded.targets)
    assert encoded.targets[-1] == tokenizer.eos_id


def test_context_expansion_preserves_existing_weights_and_positions() -> None:
    config = ModelConfig(
        vocab_size=320,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        ffn_size=64,
        dropout=0.0,
    )
    torch.manual_seed(7)
    source = CleoTransformer(config)
    checkpoint = {
        "model_config": asdict(config),
        "model_state": source.state_dict(),
    }
    expanded = expand_context_model(checkpoint, target_block_size=32, seed=99)
    assert expanded.config.block_size == 32
    assert torch.equal(
        expanded.position_embedding.weight[:16], source.position_embedding.weight
    )
    assert torch.equal(expanded.token_embedding.weight, source.token_embedding.weight)
    assert not torch.equal(
        expanded.position_embedding.weight[16:], source.position_embedding.weight
    )


def test_generalization_pipeline_smoke(tmp_path, monkeypatch) -> None:
    tokenizer = ByteBPETokenizer.train(b"general data", vocab_size=258)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    rng = np.random.default_rng(11)

    def token_file(name: str) -> str:
        path = tmp_path / name
        values = rng.integers(0, 256, size=4096, dtype=np.uint16)
        path.write_bytes(values.astype("<u2").tobytes())
        return str(path)

    story_train = token_file("story-train.bin")
    story_validation = token_file("story-validation.bin")
    general_train = token_file("general-train.bin")
    general_validation = token_file("general-validation.bin")
    general_test = token_file("general-test.bin")
    examples = [
        InstructionExample("Add one and one.", "", "Two.", "qa"),
        InstructionExample("Name a color.", "", "Blue.", "qa"),
        InstructionExample("Classify good.", "", "Positive.", "classification"),
        InstructionExample("Say hello.", "", "Hello.", "generation"),
    ]

    def instruction_file(name: str) -> str:
        path = tmp_path / name
        path.write_text(
            "".join(json.dumps(asdict(row)) + "\n" for row in examples),
            encoding="utf-8",
        )
        return str(path)

    instruction_train = instruction_file("instruction-train.jsonl")
    instruction_validation = instruction_file("instruction-validation.jsonl")
    instruction_test = instruction_file("instruction-test.jsonl")
    model_config = ModelConfig(
        vocab_size=258,
        block_size=128,
        n_layer=1,
        n_head=2,
        n_embd=32,
        ffn_size=64,
        dropout=0.0,
    )
    app_config = AppConfig(
        data=DataConfig(
            dataset_name="test/story",
            revision="0" * 40,
            license="test",
            train_url="",
            validation_url="",
            train_source="",
            validation_source="",
            train_source_bytes=0,
            validation_source_bytes=0,
            tokenizer_path=str(tokenizer_path),
            train_tokens=story_train,
            validation_tokens=story_validation,
            manifest_path=str(tmp_path / "story-manifest.json"),
            tokenizer_sample_bytes=1,
            tokenizer_vocab_size=258,
        ),
        model=model_config,
        training=TrainingConfig(
            seed=7,
            initial_microbatch_size=1,
            effective_batch_tokens=128,
            max_steps=1,
            max_wall_time_seconds=60,
            eval_interval=1,
            eval_batches=1,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=1,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            grad_clip=1.0,
            mps_memory_fraction=0.5,
            artifacts_dir=str(tmp_path),
        ),
    )
    torch.manual_seed(3)
    model = CleoTransformer(model_config)
    checkpoint_path = tmp_path / "base.pt"
    atomic_torch_save(
        {
            "format_version": 2,
            "identity": model_identity_metadata(),
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "app_config": app_config.to_dict(),
            "model_config": asdict(model_config),
            "data_manifest": {
                "dataset": "test/story",
                "revision": "0" * 40,
                "license": "test",
                "corpora": {
                    "train": {"stories": 1, "tokens": 4096},
                    "validation": {"stories": 1, "tokens": 4096},
                },
            },
            "tokenizer_checksum": ByteBPETokenizer.checksum(tokenizer_path),
            "step": 1,
            "best_validation_loss": 1.0,
            "initial_validation_loss": 2.0,
            "elapsed_training_seconds": 0.0,
        },
        checkpoint_path,
    )
    manifest_path = tmp_path / "general-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"tokenizer": {"sha256": ByteBPETokenizer.checksum(tokenizer_path)}}
        ),
        encoding="utf-8",
    )
    config = GeneralConfig(
        data=GeneralDataConfig(
            general_dataset="test/general",
            general_revision="1" * 40,
            general_license="test",
            general_config="test",
            train_urls=("x",),
            train_sources=("x",),
            train_source_bytes=(1,),
            train_source_sha256=("a" * 64,),
            validation_url="x",
            validation_source="x",
            validation_source_bytes=1,
            validation_source_sha256="b" * 64,
            test_url="x",
            test_source="x",
            test_source_bytes=1,
            test_source_sha256="c" * 64,
            instruction_dataset="test/instructions",
            instruction_revision="2" * 40,
            instruction_license="test",
            instruction_url="x",
            instruction_source="x",
            instruction_source_bytes=1,
            instruction_source_sha256="d" * 64,
            train_tokens=general_train,
            validation_tokens=general_validation,
            test_tokens=general_test,
            instruction_train=instruction_train,
            instruction_validation=instruction_validation,
            instruction_test=instruction_test,
            manifest_path=str(manifest_path),
        ),
        training=GeneralTrainingConfig(
            seed=7,
            block_size=256,
            initial_microbatch_size=1,
            effective_batch_tokens=256,
            pretrain_steps=1,
            instruction_steps=1,
            pretrain_learning_rate=1e-3,
            pretrain_min_learning_rate=1e-4,
            pretrain_warmup_steps=1,
            instruction_learning_rate=1e-3,
            weight_decay=0.01,
            grad_clip=1.0,
            story_pretrain_probability=0.0,
            instruction_batch_size=2,
            retention_batch_size=1,
            instruction_general_weight=0.1,
            instruction_story_weight=0.1,
            instruction_identity_weight=0.1,
            eval_interval=1,
            eval_batches=1,
            instruction_eval_examples=2,
            max_wall_time_seconds=60,
            required_general_loss_ratio=2.0,
            required_instruction_loss_ratio=2.0,
            max_story_loss_ratio=2.0,
            required_identity_accuracy=1.0,
        ),
    )
    identity_rows = [IdentityGeneration("identity", "canonical", True)]
    monkeypatch.setattr(general_training, "evaluate_identity_loss", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(
        general_training,
        "evaluate_identity_generation",
        lambda *args, **kwargs: identity_rows,
    )
    monkeypatch.setattr(
        general_training,
        "generate_general_responses",
        lambda *args, **kwargs: [],
    )
    output = tmp_path / "general.pt"
    report = general_training.generalize_model(
        checkpoint_path,
        output,
        tokenizer_path,
        config,
        requested_device="cpu",
        pretrain_steps=1,
        instruction_steps=1,
    )
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    assert report["completed_pretrain_steps"] == 1
    assert report["completed_instruction_steps"] == 1
    assert loaded["model_config"]["block_size"] == 256
    assert loaded["generalization"]["completed_all_steps"] is True
