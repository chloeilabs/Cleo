# Cleo 1 (`cleo-1`)

**Cleo 1** is an experimental 7.89M-parameter general-language model developed and trained by **Cleo AI** from random initialization on one Apple M4. The current alpha release broadens the original language foundation with WikiText continued pretraining, Dolly instruction tuning, a 512-token context window, and verified self-identification.

No pretrained weights or pretrained tokenizer are used. The model, lossless byte-level BPE tokenizer, causal attention, training loops, checkpoint format, API, and shadcn/ui web app are implemented in this repository.

This is a research alpha, not a reliable general-purpose assistant. It still produces repetitive text, incorrect facts, and failed instructions. See [MODEL_CARD.md](MODEL_CARD.md) before using it.

## Current release

| Item | Result |
| --- | ---: |
| Release | General-language alpha 01 |
| Parameters | 7,890,944 |
| Context | 512 tokens |
| Final training step | 23,000 |
| General validation loss | 2.0439 (51.2% below baseline) |
| General validation perplexity | 7.7204 |
| Instruction validation loss | 1.8441 (55.4% below baseline) |
| Held-out identity exact match | 8/8 |
| Foundation-distribution loss change | +17.0% |

The promoted checkpoint is `artifacts/cleo-1.pt`. The pre-generalization release is preserved as `artifacts/cleo-1-story.pt`.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --python 3.12
uv run cleo-1 --help
```

## Prepare the data

Prepare the original foundation corpus and tokenizer:

```bash
uv run cleo-1 prepare
```

Then download, verify, encode, and split the pinned general-language and instruction corpora:

```bash
uv run cleo-1 prepare-general
```

`prepare-general` verifies pinned byte sizes and SHA-256 hashes, encodes WikiText with the existing tokenizer, and creates deterministic category-stratified Dolly train, validation, and test splits. Raw and processed corpora are excluded from Git. Provenance is recorded in `data/processed/manifest.json` and `data/general/processed/manifest.json`.

## Reproduce training

The full progression is:

```bash
# 20,000-step foundation run from random weights
uv run cleo-1 train --device mps

# Initial canonical-identity adaptation
uv run cleo-1 identity-tune \
  --checkpoint artifacts/best.pt \
  --output artifacts/cleo-1.pt \
  --device mps

# 2,000 continued-pretraining steps + 600 instruction steps
uv run cleo-1 generalize \
  --checkpoint artifacts/cleo-1.pt \
  --output artifacts/cleo-1-general.pt \
  --device mps

# Repair identity, rerun retention gates, and promote only on success
uv run cleo-1 general-identity-repair \
  --checkpoint artifacts/cleo-1-general.pt \
  --device mps \
  --steps 500 \
  --promote
```

The accepted repair stopped after 100 steps. Promotion requires the general-loss, instruction-loss, foundation-retention, and 100% identity gates to pass together. An exploratory synthetic capability tune is implemented for research, but its candidate was rejected and was not promoted.

## Evaluate and generate

The standard `evaluate` command measures foundation-distribution retention. General and instruction evaluations are recorded by the gated training pipeline in the generalization reports.

```bash
uv run cleo-1 evaluate --checkpoint artifacts/cleo-1.pt --device mps

uv run cleo-1 generate \
  --checkpoint artifacts/cleo-1.pt \
  --prompt "Explain why leaves change color in autumn in two short sentences." \
  --max-new-tokens 160 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 42
```

For an accepted general-language checkpoint, CLI and web generation automatically apply the same `Instruction`/`Response` format used during tuning and return only the response. Direct identity questions return the verified canonical identity through the API.

## Local web app

```bash
uv run cleo-1 web --checkpoint artifacts/cleo-1.pt --device mps
```

The local launch site opens at `http://127.0.0.1:7860`. FastAPI loads one checkpoint instance and serves the production React, TypeScript, Tailwind CSS, and shadcn/ui build. It includes current checkpoint metrics, evaluation gates, architecture, data provenance, limitations, unedited capability probes, and a streaming playground.

Frontend development:

```bash
cd frontend
npm install
npm run dev
npm run build
```

The API exposes `/api/health`, `/api/profile`, `/api/identity`, and a cancellable NDJSON stream at `/api/generate`.

## Key artifacts

- `configs/tinystories_m4.toml` — foundation architecture and training defaults.
- `configs/general_m4.toml` — pinned generalization data, schedule, mixing, and gates.
- `artifacts/generalization_report.json` — initial generalization results.
- `artifacts/general_identity_repair_report.json` — final accepted promotion result.
- `artifacts/general_release_evaluation.json` — compact machine-readable release summary.
- `artifacts/generalization_metrics.jsonl` — per-stage training metrics.
- `artifacts/general_identity_repair_metrics.jsonl` — repair metrics.
- `artifacts/general_samples.json` — fixed, unedited qualitative probes.
- `artifacts/general_inference_benchmark.json` — warmed, five-run median MPS benchmark.
- `MODEL_CARD.md` — provenance, evaluation, intended use, and limitations.

## Tests

```bash
uv run python -m pytest
cd frontend && npm run lint && npm run build
```

The suite covers tokenizer round trips and determinism, Unicode, causal masking, parameter count, finite forward/backward passes, KV-cache parity, seeded sampling, exact CPU checkpoint continuation, identity behavior, general-data splitting and masking, context expansion, tiny end-to-end generalization, FastAPI streaming, and packaged frontend assets. MPS-specific tests run when MPS is available.

## Limitations

Cleo 1 is much broader than its original foundation checkpoint, but it is not broadly capable by modern assistant standards. At 7.89M parameters it has limited knowledge and capacity. Arithmetic, coding, multi-step reasoning, factual reliability, multilingual performance, and safety alignment are not established. Never use it for medical, legal, financial, safety-critical, or other consequential decisions.
