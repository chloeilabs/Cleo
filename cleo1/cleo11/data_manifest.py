from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .config import Cleo11Config, Cleo11DataSource


DEFAULT_MIXTURE: tuple[Cleo11DataSource, ...] = (
    Cleo11DataSource(
        name="FineWeb-Edu",
        role="pretrain_primary",
        weight=0.70,
        hub_id="HuggingFaceFW/fineweb-edu",
        hub_config="sample-10BT",
        license="ODC-By-1.0",
        notes="Educational web documents; primary quality signal for Cleo 1.1 pretraining.",
    ),
    Cleo11DataSource(
        name="FineWeb",
        role="pretrain_general",
        weight=0.15,
        hub_id="HuggingFaceFW/fineweb",
        hub_config="sample-10BT",
        license="ODC-By-1.0",
        notes="Broader web distribution to reduce FineWeb-Edu topic narrowness.",
    ),
    Cleo11DataSource(
        name="Cosmopedia v2",
        role="pretrain_synthetic",
        weight=0.10,
        hub_id="HuggingFaceTB/smollm-corpus",
        hub_config="cosmopedia-v2",
        license="ODC-By-1.0",
        notes="Synthetic educational exposition from the SmolLM corpus mixture.",
    ),
    Cleo11DataSource(
        name="Python-Edu",
        role="pretrain_code",
        weight=0.05,
        hub_id="HuggingFaceTB/smollm-corpus",
        hub_config="python-edu",
        license="ODC-By-1.0",
        text_column="text",
        content_backend="softwareheritage_s3",
        notes="Educational Python from SmolLM corpus; file bodies are fetched from Software Heritage S3.",
    ),
)


def mixture_manifest(config: Cleo11Config) -> dict[str, Any]:
    sources = config.data.sources or DEFAULT_MIXTURE
    weight_sum = sum(source.weight for source in sources)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"data mixture weights must sum to 1.0, got {weight_sum}")
    return {
        "mixture_name": config.data.mixture_name,
        "tokenizer": {
            "kind": config.data.tokenizer_kind,
            "vocab_size": config.data.tokenizer_vocab_size,
            "notes": "Train a fresh 16K byte-level BPE; do not reuse the Cleo 1 1K tokenizer.",
        },
        "minimum_pretrain_tokens": config.data.minimum_pretrain_tokens,
        "recommended_stretch_tokens": 600_000_000_000,
        "stretch_reference": "https://huggingface.co/blog/smollm",
        "stages": {
            "pretrain": "FineWeb-Edu-led mixture until token budget is met",
            "instruction": config.data.instruction_stage,
            "identity": config.data.identity_stage,
        },
        "sources": [asdict(source) for source in sources],
        "curation_rules": [
            "Pin dataset revisions, byte sizes, and SHA-256 digests before any production run.",
            "Deduplicate aggressively across sources; prefer educational quality over raw scale.",
            "Keep instruction and identity corpora out of the pretrain mixture.",
            "Record per-source token counts in the processed-data manifest after encoding.",
        ],
    }


def write_mixture_manifest(config: Cleo11Config, path: str | Path) -> dict[str, Any]:
    payload = mixture_manifest(config)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
