from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict
import json
from pathlib import Path
from threading import Lock, Timer
import time
from typing import Any, Protocol
import webbrowser

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import torch
import uvicorn

from .engine import load_trained_model, seed_everything
from .general_data import render_instruction_prompt
from .identity import (
    CANONICAL_IDENTITY_RESPONSE,
    COMPANY_NAME,
    MODEL_ID,
    MODEL_NAME,
    identity_response_for_prompt,
    model_identity_metadata,
)
from .model_profile import ModelProfile
from .tokenizer import ByteBPETokenizer


FRONTEND_DIR = Path(__file__).resolve().parent / "static"

PROMPT_STARTERS = [
    {
        "label": "Explain a concept",
        "prompt": "Explain why leaves change color in autumn in two short sentences.",
    },
    {
        "label": "Summarize text",
        "prompt": (
            "Summarize this in one sentence: Solar panels convert sunlight into "
            "electricity without burning fuel, but their output varies with weather."
        ),
    },
    {
        "label": "Extract information",
        "prompt": (
            "Extract the city and answer with only the city name: The conference "
            "will begin in Denver on Monday morning."
        ),
    },
    {
        "label": "Classify sentiment",
        "prompt": "Classify as positive or negative: The support team solved my problem quickly.",
    },
    {"label": "Model identity", "prompt": "Who are you and who trained you?"},
]


def format_model_prompt(prompt: str, *, generalized: bool) -> str:
    return render_instruction_prompt(prompt.strip(), "") if generalized else prompt


def response_from_decoded(
    decoded: str, model_prompt: str, *, generalized: bool
) -> str:
    if generalized and decoded.startswith(model_prompt):
        return decoded[len(model_prompt) :].lstrip()
    return decoded


class TextGenerator(Protocol):
    summary: str
    profile: ModelProfile

    def stream_response(
        self,
        prompt: str,
        max_new_tokens: float,
        temperature: float,
        top_k: float,
        seed: float,
    ) -> Iterator[tuple[str, str]]: ...


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    max_new_tokens: int = Field(default=300, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=1_024)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter an instruction or question first.")
        return value


class LocalTextGenerator:
    """Single in-memory model instance shared by the local FastAPI server."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: torch.device,
    ) -> None:
        self.device = device
        self.tokenizer = ByteBPETokenizer.load(tokenizer_path)
        tokenizer_checksum = ByteBPETokenizer.checksum(tokenizer_path)
        self.model, checkpoint = load_trained_model(checkpoint_path, device)
        if checkpoint["tokenizer_checksum"] != tokenizer_checksum:
            raise RuntimeError("checkpoint tokenizer checksum mismatch")
        self.profile = ModelProfile.from_runtime(
            checkpoint_path,
            checkpoint,
            parameter_count=self.model.parameter_count(),
            runtime_device=device.type,
        )
        self.summary = self.profile.summary
        self.generalized = bool(checkpoint.get("generalization", {}).get("accepted", False))
        self._generation_lock = Lock()

    def stream_response(
        self,
        prompt: str,
        max_new_tokens: float,
        temperature: float,
        top_k: float,
        seed: float,
    ) -> Iterator[tuple[str, str]]:
        if not prompt or not prompt.strip():
            raise ValueError("Enter an instruction or question first.")
        token_limit = int(max_new_tokens)
        temperature_value = float(temperature)
        top_k_value = int(top_k)
        seed_value = int(seed)
        if token_limit < 1:
            raise ValueError("Length must be at least one token.")
        if temperature_value <= 0:
            raise ValueError("Creativity must be greater than zero.")

        with self._generation_lock:
            seed_everything(seed_value)
            model_prompt = format_model_prompt(prompt, generalized=self.generalized)
            prompt_ids = self.tokenizer.encode(model_prompt, bos=True)
            tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            prompt_tokens = len(prompt_ids)
            context_note = ""
            if prompt_tokens > self.model.config.block_size:
                context_note = (
                    f" · prompt exceeded context; using its final "
                    f"{self.model.config.block_size} tokens"
                )
            yield "", f"Starting · {prompt_tokens} formatted prompt tokens{context_note}"

            started = time.perf_counter()
            generated_count = 0
            final_text = ""
            stopped_on_eos = False
            for generated in self.model.generate_steps(
                tokens,
                eos_id=self.tokenizer.eos_id,
                max_new_tokens=token_limit,
                temperature=temperature_value,
                top_k=top_k_value,
                use_cache=True,
            ):
                generated_count = generated.size(1) - prompt_tokens
                token_list = generated[0].tolist()
                decoded = self.tokenizer.decode(token_list)
                final_text = response_from_decoded(
                    decoded, model_prompt, generalized=self.generalized
                )
                stopped_on_eos = token_list[-1] == self.tokenizer.eos_id
                if generated_count == 1 or generated_count % 4 == 0 or stopped_on_eos:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    yield (
                        final_text,
                        f"Generating… {generated_count}/{token_limit} tokens · "
                        f"{generated_count / elapsed:.1f} tokens/s{context_note}",
                    )

            elapsed = max(time.perf_counter() - started, 1e-9)
            reason = "EOS emitted" if stopped_on_eos else "length limit reached"
            yield (
                final_text,
                f"Done · {generated_count} tokens · "
                f"{generated_count / elapsed:.1f} tokens/s · {reason}{context_note}",
            )


def model_profile_payload(profile: ModelProfile) -> dict[str, Any]:
    return {
        "identity": {
            "company_name": profile.company_name,
            "model_name": profile.name,
            "model_id": profile.model_id,
            "developed_and_trained_by": profile.company_name,
            "canonical_response": CANONICAL_IDENTITY_RESPONSE,
            "release": profile.release,
        },
        "runtime": {
            "device": profile.runtime_device,
            "checkpoint": profile.checkpoint_name,
            "saved_at_utc": profile.saved_at_utc,
        },
        "metrics": {
            "parameter_count": profile.parameter_count,
            "training_step": profile.training_step,
            "initial_validation_loss": profile.initial_validation_loss,
            "best_validation_loss": profile.best_validation_loss,
            "best_validation_perplexity": profile.best_validation_perplexity,
            "loss_reduction_percent": profile.loss_reduction_percent,
        },
        "architecture": {
            "block_size": profile.block_size,
            "vocab_size": profile.vocab_size,
            "n_layer": profile.n_layer,
            "n_head": profile.n_head,
            "n_embd": profile.n_embd,
            "ffn_size": profile.ffn_size,
            "dropout": profile.dropout,
        },
        "training": {
            "duration": profile.training_duration,
            "elapsed_seconds": profile.elapsed_training_seconds,
            "tokens_seen": profile.training_tokens_seen,
        },
        "adaptation": {
            "identity_tuned": profile.identity_tuned,
            "completed_steps": profile.identity_tuning_steps,
            "held_out_exact_match": profile.identity_eval_accuracy,
            "story_loss_ratio": profile.identity_story_loss_ratio,
            "deterministic_api_identity": True,
        },
        "generalization": {
            "accepted": profile.generalized,
            "foundation_steps": profile.foundation_steps,
            "foundation_identity_steps": profile.foundation_identity_steps,
            "continued_pretraining_steps": profile.general_pretrain_steps,
            "instruction_tuning_steps": profile.instruction_tuning_steps,
            "identity_repair_steps": profile.identity_repair_steps,
            "general_baseline_loss": profile.general_baseline_loss,
            "general_validation_loss": profile.general_validation_loss,
            "general_validation_perplexity": profile.general_validation_perplexity,
            "general_loss_reduction_percent": profile.general_loss_reduction_percent,
            "instruction_baseline_loss": profile.instruction_baseline_loss,
            "instruction_validation_loss": profile.instruction_validation_loss,
            "instruction_loss_reduction_percent": profile.instruction_loss_reduction_percent,
            "story_retention_ratio": profile.story_retention_ratio,
        },
        "dataset": {
            "name": profile.dataset_name,
            "revision": profile.dataset_revision,
            "license": profile.dataset_license,
            "train_stories": profile.train_stories,
            "validation_stories": profile.validation_stories,
            "train_tokens": profile.train_tokens,
            "validation_tokens": profile.validation_tokens,
            "general": {
                "name": profile.general_dataset_name,
                "revision": profile.general_dataset_revision,
                "license": profile.general_dataset_license,
                "train_documents": profile.general_train_documents,
                "validation_documents": profile.general_validation_documents,
                "train_tokens": profile.general_train_tokens,
                "validation_tokens": profile.general_validation_tokens,
            },
            "instruction": {
                "name": profile.instruction_dataset_name,
                "revision": profile.instruction_dataset_revision,
                "license": profile.instruction_dataset_license,
                "train_examples": profile.instruction_train_examples,
                "validation_examples": profile.instruction_validation_examples,
                "test_examples": profile.instruction_test_examples,
            },
        },
        "benchmark": {
            "scope": (
                "loaded general-language checkpoint"
                if profile.generalized
                else "loaded checkpoint"
            ),
            "device": profile.benchmark_device,
            "cached_tokens_per_second": profile.cached_tokens_per_second,
            "uncached_tokens_per_second": profile.uncached_tokens_per_second,
            "cache_speedup": profile.cache_speedup,
            "new_tokens": profile.benchmark_new_tokens,
            "outputs_equal": profile.benchmark_outputs_equal,
        },
        "validation_curve": [asdict(point) for point in profile.validation_curve],
        "samples": [asdict(sample) for sample in profile.samples],
        "prompt_starters": PROMPT_STARTERS,
    }


async def _generation_events(
    generator: TextGenerator, request: GenerationRequest
) -> AsyncIterator[bytes]:
    identity_response = identity_response_for_prompt(request.prompt)
    if identity_response is not None:
        for text, status in (
            ("", "Recognized model identity question"),
            (
                identity_response,
                "Done · verified checkpoint identity · no sampling required",
            ),
        ):
            event = {"type": "generation", "text": text, "status": status}
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        return

    stream = generator.stream_response(
        request.prompt,
        request.max_new_tokens,
        request.temperature,
        request.top_k,
        request.seed,
    )
    try:
        for text, status in stream:
            event = {"type": "generation", "text": text, "status": status}
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            # Give Starlette a cancellation point after every streamed update.
            # Closing this async wrapper also closes the underlying model iterator,
            # which releases LocalTextGenerator's generation lock immediately.
            await asyncio.sleep(0)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


def create_app(
    generator: TextGenerator,
    *,
    static_dir: str | Path = FRONTEND_DIR,
) -> FastAPI:
    app = FastAPI(
        title=f"{COMPANY_NAME} — {MODEL_NAME} API",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "company": COMPANY_NAME,
            "model": MODEL_NAME,
            "model_id": MODEL_ID,
        }

    @app.get("/api/profile")
    def profile() -> dict[str, Any]:
        return model_profile_payload(generator.profile)

    @app.get("/api/identity")
    def identity() -> dict[str, Any]:
        return model_identity_metadata()

    @app.post("/api/generate", response_class=StreamingResponse)
    def generate(request: GenerationRequest) -> StreamingResponse:
        return StreamingResponse(
            _generation_events(generator, request),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )

    frontend = Path(static_dir).resolve()
    index = frontend / "index.html"
    if index.is_file():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        def frontend_missing() -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Frontend build is missing. Run `npm run build` in frontend/.",
                },
            )

    return app


def launch_web(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    *,
    device: torch.device,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    generator = LocalTextGenerator(checkpoint_path, tokenizer_path, device)
    print(f"Loaded {generator.summary}", flush=True)
    app = create_app(generator)
    if not (FRONTEND_DIR / "index.html").is_file():
        raise FileNotFoundError(
            f"frontend build not found at {FRONTEND_DIR}; run `npm run build` in frontend/"
        )
    if open_browser:
        timer = Timer(0.8, webbrowser.open, args=(f"http://{host}:{port}",))
        timer.daemon = True
        timer.start()
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="info")
