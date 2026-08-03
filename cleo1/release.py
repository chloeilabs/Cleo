from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ALPHA_TAG = "cleo-1-general-alpha-01"
DEFAULT_CHECKPOINT = Path("artifacts/cleo-1.pt")
DEFAULT_TOKENIZER = Path("data/processed/tokenizer.json")
DEFAULT_EVALUATION = Path("artifacts/general_release_evaluation.json")
DEFAULT_MODEL_CARD = Path("MODEL_CARD.md")
DEFAULT_RELEASE_DIR = Path("releases") / ALPHA_TAG


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_alpha_release(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    tokenizer: Path = DEFAULT_TOKENIZER,
    evaluation: Path = DEFAULT_EVALUATION,
    model_card: Path = DEFAULT_MODEL_CARD,
    release_dir: Path = DEFAULT_RELEASE_DIR,
    frozen_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Freeze Cleo 1 general-language alpha 01 packaging metadata and sidecar files.

    Weights stay out of git (see ``.gitignore``) and are published as release assets.
    """

    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer}")
    if not evaluation.is_file():
        raise FileNotFoundError(f"evaluation summary not found: {evaluation}")
    if not model_card.is_file():
        raise FileNotFoundError(f"model card not found: {model_card}")

    release_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = frozen_checkpoint or Path("artifacts") / f"{ALPHA_TAG}.pt"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, frozen_path)

    tokenizer_dest = release_dir / "tokenizer.json"
    evaluation_dest = release_dir / "general_release_evaluation.json"
    model_card_dest = release_dir / "MODEL_CARD.md"
    shutil.copy2(tokenizer, tokenizer_dest)
    shutil.copy2(evaluation, evaluation_dest)
    shutil.copy2(model_card, model_card_dest)

    checkpoint_sha = sha256_file(frozen_path)
    tokenizer_sha = sha256_file(tokenizer_dest)
    evaluation_payload = json.loads(evaluation_dest.read_text(encoding="utf-8"))
    if evaluation_payload.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(
            "evaluation checkpoint_sha256 does not match the frozen checkpoint; "
            f"expected {checkpoint_sha}, found {evaluation_payload.get('checkpoint_sha256')}"
        )

    payload = {
        "tag": ALPHA_TAG,
        "model_name": "Cleo 1",
        "model_id": "cleo-1",
        "release": "General-language alpha 01",
        "status": "frozen",
        "successor": "cleo-1.1",
        "rationale": (
            "Cleo 1 is frozen at 7.89M parameters, 1,024-token vocabulary, and "
            "512-token context. Further fine-tuning cannot overcome those capacity "
            "limits; Cleo 1.1 is the successor architecture."
        ),
        "checkpoint": {
            "artifact_name": f"{ALPHA_TAG}.pt",
            "local_path": str(frozen_path),
            "source_path": str(checkpoint),
            "sha256": checkpoint_sha,
            "parameter_count": evaluation_payload.get("parameter_count"),
            "context_tokens": evaluation_payload.get("context_tokens"),
            "vocab_size": 1024,
            "final_step": evaluation_payload.get("stages", {}).get("total_step"),
            "format_version": 3,
        },
        "tokenizer": {
            "artifact_name": "tokenizer.json",
            "sha256": tokenizer_sha,
            "kind": "byte_bpe",
            "vocab_size": 1024,
        },
        "bundled_artifacts": [
            "MODEL_CARD.md",
            "general_release_evaluation.json",
            "tokenizer.json",
            "RELEASE.json",
        ],
        "weights_distribution": "GitHub Release assets (not committed to git)",
        "promotion_note": (
            "Do not promote Cleo 1 further on capability grounds. "
            "Use this checkpoint only as the frozen alpha baseline."
        ),
        "evaluation": evaluation_payload,
    }
    release_json = release_dir / "RELEASE.json"
    release_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def github_release_asset_paths(
    *,
    frozen_checkpoint: Path | None = None,
    release_dir: Path = DEFAULT_RELEASE_DIR,
) -> list[Path]:
    frozen_path = frozen_checkpoint or Path("artifacts") / f"{ALPHA_TAG}.pt"
    return [
        frozen_path,
        release_dir / "tokenizer.json",
        release_dir / "MODEL_CARD.md",
        release_dir / "general_release_evaluation.json",
        release_dir / "RELEASE.json",
    ]
