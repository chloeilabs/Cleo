from __future__ import annotations

import json

import torch

from cleo1.cleo11.compute import estimate_compute
from cleo1.cleo11.config import Cleo11ModelConfig, load_cleo11_config
from cleo1.cleo11.data_manifest import mixture_manifest
from cleo1.cleo11.evaluation import evaluate_gates, evaluation_contract
from cleo1.cleo11.model import Cleo11Transformer
from cleo1.cleo11.train import run_smoke_train, write_training_spec
from cleo1.release import package_alpha_release


def tiny_cleo11_config() -> Cleo11ModelConfig:
    return Cleo11ModelConfig(
        vocab_size=320,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        ffn_size=128,
        dropout=0.0,
    )


def test_cleo11_135m_parameter_count_is_near_target():
    config = load_cleo11_config("configs/cleo11_135m.toml")
    model = Cleo11Transformer(config.model)
    count = model.parameter_count()
    assert 130_000_000 <= count <= 140_000_000
    assert count == 135_862_400


def test_cleo11_causal_mask_and_cache_parity():
    torch.manual_seed(7)
    model = Cleo11Transformer(tiny_cleo11_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 90, 91, 92, 93]])
    first_logits, _ = model(first)
    second_logits, _ = model(second)
    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2], rtol=0, atol=0)

    tokens = torch.randint(0, 320, (2, 17))
    full_logits, _ = model(tokens)
    first_logits, cache = model.forward_with_cache(tokens[:, :5])
    second_logits, cache = model.forward_with_cache(tokens[:, 5:12], cache)
    third_logits, cache = model.forward_with_cache(tokens[:, 12:], cache)
    cached = torch.cat((first_logits, second_logits, third_logits), dim=1)
    torch.testing.assert_close(cached, full_logits, rtol=1e-5, atol=1e-6)
    assert cache[0][0].shape[:3] == (2, model.config.n_kv_head, 17)


def test_cleo11_compute_and_contracts():
    config = load_cleo11_config("configs/cleo11_135m.toml")
    compute = estimate_compute(config, parameter_count=135_862_400)
    assert compute.meets_chinchilla_floor
    assert compute.target_tokens >= 2_700_000_000
    assert compute.target_tokens == 2_720_000_000
    mixture = mixture_manifest(config)
    assert abs(sum(source["weight"] for source in mixture["sources"]) - 1.0) < 1e-9
    contract = evaluation_contract(config)
    assert {gate["category"] for gate in contract["categories"]} >= {
        "reasoning",
        "arithmetic",
        "extraction",
        "knowledge",
        "code",
        "safety",
        "instruction_following",
        "identity",
    }
    report = evaluate_gates(
        config,
        {
            "reasoning": 0.56,
            "arithmetic": 0.71,
            "extraction": 0.66,
            "knowledge": 0.36,
            "code": 0.41,
            "safety": 0.91,
            "instruction_following": 0.61,
            "identity": 1.0,
        },
    )
    assert report.accepted
    failing = evaluate_gates(
        config,
        {
            "reasoning": 0.10,
            "arithmetic": 0.71,
            "extraction": 0.66,
            "knowledge": 0.36,
            "code": 0.41,
            "safety": 0.91,
            "instruction_following": 0.61,
            "identity": 1.0,
        },
    )
    assert not failing.accepted


def test_cleo11_smoke_train_end_to_end(tmp_path):
    config = load_cleo11_config("configs/cleo11_smoke.toml")
    report = run_smoke_train(config, requested_device="cpu", output_dir=tmp_path)
    assert report["accepted_smoke"]
    assert (tmp_path / "cleo11-smoke.pt").is_file()
    assert (tmp_path / "training_spec.json").is_file()
    assert (tmp_path / "dataset_manifest.json").is_file()
    assert (tmp_path / "evaluation_contract.json").is_file()
    spec = json.loads((tmp_path / "training_spec.json").read_text(encoding="utf-8"))
    assert "RoPE" in spec["architecture"]["features"]


def test_package_alpha_release(tmp_path):
    checkpoint = tmp_path / "cleo-1.pt"
    tokenizer = tmp_path / "tokenizer.json"
    evaluation = tmp_path / "eval.json"
    model_card = tmp_path / "MODEL_CARD.md"
    release_dir = tmp_path / "release"
    frozen = tmp_path / "frozen.pt"
    torch.save({"format_version": 3, "step": 23000}, checkpoint)
    tokenizer.write_text('{"format_version": 1, "kind": "byte_bpe"}\n', encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evaluation.write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "parameter_count": 7890944,
                "context_tokens": 512,
                "stages": {"total_step": 23000},
            }
        ),
        encoding="utf-8",
    )
    model_card.write_text("# card\n", encoding="utf-8")
    payload = package_alpha_release(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        evaluation=evaluation,
        model_card=model_card,
        release_dir=release_dir,
        frozen_checkpoint=frozen,
    )
    assert payload["tag"] == "cleo-1-general-alpha-01"
    assert payload["status"] == "frozen"
    assert frozen.is_file()
    assert (release_dir / "RELEASE.json").is_file()


def test_write_training_spec_for_135m(tmp_path):
    config = load_cleo11_config("configs/cleo11_135m.toml")
    payload = write_training_spec(config, tmp_path)
    assert payload["architecture"]["parameter_count"] == 135_862_400
    assert payload["compute"]["meets_chinchilla_floor"]
