from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch

from .checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint, restore_rng_state
from .config import AppConfig, ModelConfig
from .data import TokenCorpus
from .identity import COMPANY_NAME, MODEL_ID, MODEL_NAME, model_identity_metadata
from .model import CleoTransformer
from .tokenizer import ByteBPETokenizer


FIXED_PROMPTS = [
    "Once upon a time, there was a little fox",
    "Lily found a shiny key in the garden",
    "The small blue bird wanted to learn how to sing",
    "One rainy morning, Tom and his dog",
    "Mia was afraid of the dark, but then",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except RuntimeError:
        pass


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch runtime")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")


def configure_device(device: torch.device, config: AppConfig) -> None:
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(config.training.mps_memory_fraction)


def build_optimizer(model: CleoTransformer, config: AppConfig) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.training.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.epsilon,
    )


def learning_rate_for_step(step: int, config: AppConfig) -> float:
    training = config.training
    if step < training.warmup_steps:
        return training.learning_rate * (step + 1) / training.warmup_steps
    if step >= training.max_steps:
        return training.min_learning_rate
    progress = (step - training.warmup_steps) / max(training.max_steps - training.warmup_steps, 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return training.min_learning_rate + coefficient * (
        training.learning_rate - training.min_learning_rate
    )


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "not enough memory" in message


def choose_batch_plan(
    model: CleoTransformer,
    corpus: TokenCorpus,
    device: torch.device,
    config: AppConfig,
) -> tuple[int, int]:
    target_sequences = config.training.effective_batch_tokens // config.model.block_size
    if target_sequences * config.model.block_size != config.training.effective_batch_tokens:
        raise ValueError("effective_batch_tokens must be divisible by block_size")
    microbatch = min(config.training.initial_microbatch_size, target_sequences)
    while microbatch >= 1:
        if target_sequences % microbatch:
            microbatch //= 2
            continue
        generator = torch.Generator(device="cpu").manual_seed(config.training.seed + 91)
        try:
            model.train()
            model.zero_grad(set_to_none=True)
            inputs, targets = corpus.random_batch(microbatch, generator, device)
            _, loss = model(inputs, targets)
            assert loss is not None
            loss.backward()
            if device.type == "mps":
                torch.mps.synchronize()
            model.zero_grad(set_to_none=True)
            return microbatch, target_sequences // microbatch
        except RuntimeError as error:
            model.zero_grad(set_to_none=True)
            if not _is_oom(error):
                raise
            if device.type == "mps":
                torch.mps.empty_cache()
            print(f"Microbatch {microbatch} exceeded device memory; retrying smaller", flush=True)
            microbatch //= 2
    raise RuntimeError("unable to fit even one training sequence on the selected device")


@torch.no_grad()
def evaluate_model(
    model: CleoTransformer,
    corpus: TokenCorpus,
    device: torch.device,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        inputs, targets = corpus.random_batch(batch_size, generator, device)
        _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def _append_metric(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _checkpoint_payload(
    *,
    model: CleoTransformer,
    optimizer: torch.optim.AdamW,
    config: AppConfig,
    manifest: dict[str, Any],
    tokenizer_checksum: str,
    data_generator: torch.Generator,
    step: int,
    best_validation_loss: float,
    initial_validation_loss: float,
    elapsed_training_seconds: float,
    microbatch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "identity": model_identity_metadata(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "app_config": config.to_dict(),
        "model_config": asdict(config.model),
        "data_manifest": manifest,
        "tokenizer_checksum": tokenizer_checksum,
        "rng_state": capture_rng_state(data_generator),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "initial_validation_loss": initial_validation_loss,
        "elapsed_training_seconds": elapsed_training_seconds,
        "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


@dataclass(frozen=True)
class TrainingResult:
    step: int
    initial_validation_loss: float
    best_validation_loss: float
    final_validation_loss: float
    elapsed_training_seconds: float
    stopping_reason: str
    checkpoint: str
    acceptance_passed: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["initial_validation_perplexity"] = _perplexity(self.initial_validation_loss)
        value["best_validation_perplexity"] = _perplexity(self.best_validation_loss)
        value["final_validation_perplexity"] = _perplexity(self.final_validation_loss)
        value["required_relative_loss"] = 0.75
        return value


def train_model(
    config: AppConfig,
    *,
    requested_device: str = "auto",
    resume_path: str | None = None,
) -> TrainingResult:
    manifest_path = Path(config.data.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"prepared-data manifest is missing: {manifest_path}; run prepare first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_checksum = ByteBPETokenizer.checksum(config.data.tokenizer_path)
    if tokenizer_checksum != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("tokenizer checksum does not match the prepared-data manifest")
    train_corpus = TokenCorpus(config.data.train_tokens, config.model.block_size)
    validation_corpus = TokenCorpus(config.data.validation_tokens, config.model.block_size)
    device = select_device(requested_device)
    configure_device(device, config)
    seed_everything(config.training.seed)
    model = CleoTransformer(config.model).to(device)
    optimizer = build_optimizer(model, config)
    print(
        f"Device={device.type} parameters={model.parameter_count():,} "
        f"train_tokens={len(train_corpus):,} validation_tokens={len(validation_corpus):,}",
        flush=True,
    )
    microbatch, accumulation_steps = choose_batch_plan(model, train_corpus, device, config)
    print(
        f"Training batch plan: microbatch={microbatch}, accumulation={accumulation_steps}, "
        f"effective_tokens={microbatch * accumulation_steps * config.model.block_size:,}",
        flush=True,
    )
    data_generator = torch.Generator(device="cpu").manual_seed(config.training.seed + 1)
    step = 0
    best_loss = math.inf
    initial_loss = math.nan
    base_elapsed = 0.0
    if resume_path:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        if checkpoint["app_config"] != config.to_dict():
            raise RuntimeError("resume checkpoint configuration does not match the requested configuration")
        if checkpoint["tokenizer_checksum"] != tokenizer_checksum:
            raise RuntimeError("resume checkpoint tokenizer checksum mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = int(checkpoint["step"])
        best_loss = float(checkpoint["best_validation_loss"])
        initial_loss = float(checkpoint["initial_validation_loss"])
        base_elapsed = float(checkpoint.get("elapsed_training_seconds", 0.0))
        restore_rng_state(checkpoint["rng_state"], data_generator)
        print(f"Resumed {resume_path} at step {step:,}", flush=True)
    else:
        seed_everything(config.training.seed)
        data_generator.manual_seed(config.training.seed + 1)

    artifacts = Path(config.training.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    latest_path = artifacts / "latest.pt"
    best_path = artifacts / "best.pt"
    metrics_path = artifacts / "metrics.jsonl"
    if not resume_path and metrics_path.exists():
        metrics_path.unlink()
    run_started = time.monotonic()

    def elapsed() -> float:
        return base_elapsed + (time.monotonic() - run_started)

    def save(path: Path) -> None:
        atomic_torch_save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                manifest=manifest,
                tokenizer_checksum=tokenizer_checksum,
                data_generator=data_generator,
                step=step,
                best_validation_loss=best_loss,
                initial_validation_loss=initial_loss,
                elapsed_training_seconds=elapsed(),
                microbatch_size=microbatch,
                gradient_accumulation_steps=accumulation_steps,
            ),
            path,
        )

    if not resume_path:
        print("Measuring initialization validation baseline", flush=True)
        initial_loss = evaluate_model(
            model,
            validation_corpus,
            device,
            batches=config.training.eval_batches,
            batch_size=microbatch,
            seed=config.training.seed + 2,
        )
        best_loss = initial_loss
        metric = {
            "event": "validation",
            "step": 0,
            "loss": initial_loss,
            "perplexity": _perplexity(initial_loss),
            "elapsed_seconds": elapsed(),
        }
        _append_metric(metrics_path, metric)
        print(f"step=0 validation_loss={initial_loss:.4f} perplexity={_perplexity(initial_loss):.2f}", flush=True)
        save(best_path)
        save(latest_path)

    stopping_reason = "max_steps"
    last_eval_step = 0 if not resume_path else -1
    log_started = time.monotonic()
    log_step = step
    try:
        while step < config.training.max_steps:
            if elapsed() >= config.training.max_wall_time_seconds:
                stopping_reason = "wall_time_limit"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                inputs, targets = train_corpus.random_batch(microbatch, data_generator, device)
                _, loss = model(inputs, targets)
                assert loss is not None
                if not bool(torch.isfinite(loss).item()):
                    stopping_reason = "non_finite_loss"
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                accumulated_loss += float(loss.detach().item())
                (loss / accumulation_steps).backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            if not bool(torch.isfinite(gradient_norm).item()):
                stopping_reason = "non_finite_gradient"
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            lr = learning_rate_for_step(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step += 1
            train_loss = accumulated_loss / accumulation_steps

            if step % 50 == 0:
                if device.type == "mps":
                    torch.mps.synchronize()
                now = time.monotonic()
                completed = step - log_step
                tokens_per_second = (
                    completed * config.training.effective_batch_tokens / max(now - log_started, 0.001)
                )
                print(
                    f"step={step:,}/{config.training.max_steps:,} loss={train_loss:.4f} "
                    f"lr={lr:.2e} tok/s={tokens_per_second:,.0f} elapsed={elapsed() / 3600:.2f}h",
                    flush=True,
                )
                _append_metric(
                    metrics_path,
                    {
                        "event": "train",
                        "step": step,
                        "loss": train_loss,
                        "learning_rate": lr,
                        "tokens_per_second": tokens_per_second,
                        "gradient_norm": float(gradient_norm.item()),
                        "elapsed_seconds": elapsed(),
                    },
                )
                log_started = now
                log_step = step

            if step % config.training.eval_interval == 0:
                validation_loss = evaluate_model(
                    model,
                    validation_corpus,
                    device,
                    batches=config.training.eval_batches,
                    batch_size=microbatch,
                    seed=config.training.seed + 2,
                )
                last_eval_step = step
                improved = validation_loss < best_loss
                if improved:
                    best_loss = validation_loss
                _append_metric(
                    metrics_path,
                    {
                        "event": "validation",
                        "step": step,
                        "loss": validation_loss,
                        "perplexity": _perplexity(validation_loss),
                        "best": improved,
                        "elapsed_seconds": elapsed(),
                    },
                )
                print(
                    f"step={step:,} validation_loss={validation_loss:.4f} "
                    f"perplexity={_perplexity(validation_loss):.2f} best={best_loss:.4f}",
                    flush=True,
                )
                save(latest_path)
                if improved:
                    save(best_path)
    except KeyboardInterrupt:
        stopping_reason = "keyboard_interrupt"
        print("Training interrupted; saving latest checkpoint", flush=True)
        save(latest_path)
    except FloatingPointError:
        print("Training stopped on a non-finite value; the last valid checkpoint is preserved", flush=True)
        raise

    print("Running final validation", flush=True)
    final_loss = evaluate_model(
        model,
        validation_corpus,
        device,
        batches=config.training.eval_batches,
        batch_size=microbatch,
        seed=config.training.seed + 2,
    )
    improved = final_loss < best_loss
    if improved:
        best_loss = final_loss
    _append_metric(
        metrics_path,
        {
            "event": "validation_final",
            "step": step,
            "loss": final_loss,
            "perplexity": _perplexity(final_loss),
            "best": improved,
            "elapsed_seconds": elapsed(),
        },
    )
    save(latest_path)
    if improved:
        save(best_path)
    result = TrainingResult(
        step=step,
        initial_validation_loss=initial_loss,
        best_validation_loss=best_loss,
        final_validation_loss=final_loss,
        elapsed_training_seconds=elapsed(),
        stopping_reason=stopping_reason,
        checkpoint=str(best_path),
        acceptance_passed=best_loss <= initial_loss * 0.75,
    )
    (artifacts / "training_report.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)
    return result


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[CleoTransformer, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = CleoTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def generate_text(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    *,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    min_new_tokens: int = 0,
    use_cache: bool = True,
) -> str:
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    expected_checksum = ByteBPETokenizer.checksum(tokenizer_path)
    model, checkpoint = load_trained_model(checkpoint_path, device)
    if checkpoint["tokenizer_checksum"] != expected_checksum:
        raise RuntimeError("checkpoint tokenizer checksum mismatch")
    seed_everything(seed)
    generalized = bool(checkpoint.get("generalization", {}).get("accepted", False))
    if generalized:
        from .general_data import render_instruction_prompt

        model_prompt = render_instruction_prompt(prompt.strip(), "")
    else:
        model_prompt = prompt
    prompt_ids = tokenizer.encode(model_prompt, bos=True)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        tokens,
        eos_id=tokenizer.eos_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        min_new_tokens=min_new_tokens,
        use_cache=use_cache,
    )
    decoded = tokenizer.decode(generated[0].tolist())
    if generalized and decoded.startswith(model_prompt):
        return decoded[len(model_prompt) :].lstrip()
    return decoded


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    requested_device: str,
    batches: int | None = None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = AppConfig.from_dict(checkpoint["app_config"])
    device = select_device(requested_device)
    configure_device(device, config)
    model, _ = load_trained_model(checkpoint_path, device)
    corpus = TokenCorpus(config.data.validation_tokens, config.model.block_size)
    batch_size = int(checkpoint.get("microbatch_size", config.training.initial_microbatch_size))
    loss = evaluate_model(
        model,
        corpus,
        device,
        batches=batches or config.training.eval_batches,
        batch_size=batch_size,
        seed=config.training.seed + 2,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint["step"]),
        "validation_loss": loss,
        "validation_perplexity": _perplexity(loss),
        "device": device.type,
        "batches": batches or config.training.eval_batches,
    }


def write_fixed_samples(
    config: AppConfig, checkpoint_path: str | Path
) -> list[dict[str, Any]]:
    artifacts = Path(config.training.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(config.data.tokenizer_path)
    samples: list[dict[str, Any]] = []
    print("Generating five deterministic CPU samples", flush=True)
    for index, prompt in enumerate(FIXED_PROMPTS):
        text = generate_text(
            checkpoint_path,
            tokenizer_path,
            prompt=prompt,
            device=torch.device("cpu"),
            max_new_tokens=300,
            temperature=0.8,
            top_k=40,
            seed=config.training.seed + 100 + index,
            min_new_tokens=100,
        )
        samples.append({"prompt": prompt, "seed": config.training.seed + 100 + index, "text": text})
    (artifacts / "samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = ["# Fixed-prompt samples", ""]
    for sample in samples:
        markdown.extend([f"## {sample['prompt']}", "", sample["text"], ""])
    (artifacts / "samples.md").write_text("\n".join(markdown), encoding="utf-8")
    return samples


def write_final_artifacts(config: AppConfig, checkpoint_path: str | Path) -> None:
    artifacts = Path(config.training.artifacts_dir)
    write_fixed_samples(config, checkpoint_path)
    report_path = artifacts / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    checkpoint = load_checkpoint(checkpoint_path)
    adaptation = dict(checkpoint.get("adaptation", {}))
    identity_results = ""
    if adaptation.get("accepted"):
        identity_results = f"""
## Identity adaptation

- Fine-tuning steps: {adaptation.get('completed_steps')}
- Held-out exact match: {float(adaptation.get('final_identity_accuracy', 0.0)):.0%}
- Paired story-loss change: {(float(adaptation.get('story_loss_ratio', 1.0)) - 1.0) * 100:.2f}%
- Canonical response: {model_identity_metadata()['canonical_response']}

This is a narrow memorized identity behavior, not evidence of self-awareness or general instruction following.
"""
    model_card = f"""# {MODEL_NAME} — Model Card

## Model

{MODEL_NAME} (`{MODEL_ID}`) is a {CleoTransformer(config.model).parameter_count():,}-parameter decoder-only transformer developed and trained by {COMPANY_NAME} from random initialization. It uses a custom 1,024-token byte-level BPE vocabulary, a 256-token context, six transformer blocks, five attention heads, and 320-dimensional embeddings.

## Training data

The model uses the first official training Parquet shard and the validation split of `roneneldan/TinyStories`, pinned to revision `{config.data.revision}`. The dataset is distributed under {config.data.license}. No pretrained weights or pretrained tokenizer were used.

## Results

- Training steps: {report.get('step', 'not recorded')}
- Initial validation loss: {report.get('initial_validation_loss', 'not recorded')}
- Best validation loss: {report.get('best_validation_loss', 'not recorded')}
- Best validation perplexity: {report.get('best_validation_perplexity', 'not recorded')}
- Acceptance gate passed: {report.get('acceptance_passed', False)}
- Stopping reason: {report.get('stopping_reason', 'not recorded')}

See `samples.md` for fixed-prompt generations and `metrics.jsonl` for the training history.
{identity_results}

## Intended use and limitations

This model is an educational, narrow story generator. It is not a chatbot, factual knowledge source, or safety system. Its stories may be repetitive, inconsistent, biased, or unsuitable. Do not use it for factual or safety-critical decisions.
"""
    (artifacts / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
