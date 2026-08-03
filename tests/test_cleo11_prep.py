from __future__ import annotations

import json
from pathlib import Path

from cleo1.cleo11.config import load_cleo11_config
from cleo1.cleo11.corpus import ShardedTokenCorpus
from cleo1.cleo11.launch import emit_launch_plan, write_launch_script
from cleo1.cleo11.prepare import prepare_cleo11_data
from cleo1.cleo11.pretrain import pretrain_cleo11


def _smoke_config(tmp_path: Path):
    config = load_cleo11_config("configs/cleo11_smoke.toml")
    prep = config.prep
    output = tmp_path / "processed"
    return type(config)(
        model=config.model,
        training=type(config.training)(
            **{
                **config.training.__dict__,
                "artifacts_dir": str(tmp_path / "artifacts"),
                "eval_interval": 2,
                "eval_batches": 2,
                "warmup_steps": 1,
                "effective_batch_tokens": 256,
                "initial_microbatch_size": 2,
            }
        ),
        data=config.data,
        evaluation=config.evaluation,
        prep=type(prep)(
            **{
                **prep.__dict__,
                "output_dir": str(output),
                "tokenizer_path": str(output / "tokenizer.json"),
                "train_glob": str(output / "train-*.bin"),
                "validation_tokens_path": str(output / "validation.bin"),
                "manifest_path": str(output / "manifest.json"),
                "shard_tokens": 4096,
                "default_profile": "dev",
            }
        ),
    )


def test_synthetic_prepare_and_sharded_corpus(tmp_path):
    config = _smoke_config(tmp_path)
    manifest = prepare_cleo11_data(config, profile_name="dev", synthetic=True, force=True)
    assert manifest["synthetic"] is True
    assert manifest["corpora"]["train"]["tokens"] >= 8192
    assert manifest["corpora"]["validation"]["tokens"] >= 2048
    assert Path(manifest["tokenizer"]["path"]).is_file()
    corpus = ShardedTokenCorpus(manifest["corpora"]["train"]["paths"], config.model.block_size)
    assert len(corpus) == manifest["corpora"]["train"]["tokens"]


def test_synthetic_prepare_then_short_pretrain(tmp_path):
    config = _smoke_config(tmp_path)
    prepare_cleo11_data(config, profile_name="dev", synthetic=True, force=True)
    result = pretrain_cleo11(
        config,
        requested_device="cpu",
        max_steps_override=3,
        microbatch_override=2,
    )
    assert result.steps == 3
    assert Path(result.checkpoint).is_file()
    report = json.loads((tmp_path / "artifacts" / "pretrain_report.json").read_text(encoding="utf-8"))
    assert report["accepted_pretrain_run"] is True


def test_launch_plan_dry_run_and_script(tmp_path):
    plan = emit_launch_plan(profile="full", nproc=2, max_steps=10)
    assert plan["dry_run"] is True
    assert "docker build" in plan["commands"]["docker_build"]
    assert "--gpus all" in plan["commands"]["docker_run"]
    assert "torchrun" in plan["commands"]["native_torchrun"]
    script = write_launch_script(tmp_path / "launch.sh", plan)
    text = script.read_text(encoding="utf-8")
    assert "docker build" in text
    assert "docker run" in text


def test_135m_config_has_pinned_hub_configs():
    config = load_cleo11_config("configs/cleo11_135m.toml")
    by_name = {source.name: source for source in config.data.sources}
    assert by_name["FineWeb-Edu"].hub_config == "sample-10BT"
    assert by_name["FineWeb"].hub_config == "sample-10BT"
    assert by_name["Cosmopedia v2"].hub_id == "HuggingFaceTB/smollm-corpus"
    assert by_name["Python-Edu"].hub_config == "python-edu"
    assert by_name["Python-Edu"].content_backend == "softwareheritage_s3"
    assert config.prep.profile("full").train_tokens == 2_720_000_000
    assert config.prep.profile("dev").train_tokens == 50_000_000
