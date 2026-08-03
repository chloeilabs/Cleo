# Cleo 1 (`cleo-1`)

**Cleo AI** is the app and company that developed and trained **Cleo 1** (`cleo-1`), a decoder-only story model trained from random weights on the TinyStories dataset. The project includes a custom byte-level BPE tokenizer, explicit causal self-attention, resumable training, validation metrics, a command-line story generator, and a local browser interface. No pretrained model or tokenizer is used.

The default configuration is sized for an Apple M4 Mac with 16 GB unified memory:

- 7,809,024 trainable parameters
- 6 transformer blocks, 320 embedding dimensions, 5 attention heads
- 1,024-token custom byte-BPE vocabulary and 256-token context
- FP32 PyTorch training through the MPS backend
- 20,000-step or four-hour stopping limit

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --python 3.12
uv run cleo-1 --help
```

## Prepare the data

The preparation command downloads the first official TinyStories training shard and the full validation shard at the pinned revision in `configs/tinystories_m4.toml`. It verifies file sizes, records SHA-256 hashes, trains the tokenizer, and writes little-endian `uint16` token streams.

```bash
uv run cleo-1 prepare
```

The raw and processed data are intentionally excluded from Git. Re-running `prepare` is idempotent; use `--force` only to rebuild processed artifacts.

## Train

```bash
uv run cleo-1 train --device mps
```

The trainer selects the largest microbatch that fits while preserving an effective 8,192 tokens per optimizer step. It writes `artifacts/latest.pt`, updates `artifacts/best.pt` when validation improves, and can resume with:

```bash
uv run cleo-1 train --device mps --resume artifacts/latest.pt
```

Checkpoints include model and optimizer state, configuration, data manifest, tokenizer checksum, RNG states, elapsed training time, and validation history. The four-hour limit is cumulative across resumes.

## Teach the release identity

The story-quality checkpoint remains `artifacts/best.pt`. A bounded supervised continuation teaches the canonical release identity and writes the gated release checkpoint to `artifacts/cleo-1.pt`:

```bash
uv run cleo-1 identity-tune --checkpoint artifacts/best.pt --device mps
```

The command trains on varied identity prompts, evaluates unseen paraphrases with greedy decoding, and uses a 4:1 story-to-identity loss mix on every update. It stops only when all held-out prompts exactly reproduce the canonical identity, paired story-validation loss remains within the configured retention limit, and full-length seeded story probes contain no identity leakage. The accepted run took 300 steps, reached 100% exact match on 8 held-out prompts, and changed paired story-validation loss by 0.13%. Details are recorded in `artifacts/identity_finetune_report.json` and `artifacts/identity_metrics.jsonl`.

## Evaluate and generate

```bash
uv run cleo-1 evaluate --checkpoint artifacts/cleo-1.pt --device mps
uv run cleo-1 generate \
  --checkpoint artifacts/cleo-1.pt \
  --prompt "Once upon a time, there was a little fox" \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 42
```

Generation uses a per-layer key/value attention cache by default. The cache does not alter the model or checkpoint weights, and it is rebuilt exactly when the learned 256-position context window rolls over. Use `--no-kv-cache` only to compare against the slower full-forward path.

## Local web interface

Launch the trained model on the M4 GPU:

```bash
uv run cleo-1 web --checkpoint artifacts/cleo-1.pt --device mps
```

The `cleo-1` command loads the checkpoint once and opens the model-launch site at `http://127.0.0.1:7860`. FastAPI serves the local model API and the production React build; the interface is built with TypeScript, Vite, Tailwind CSS, and shadcn/ui. The site includes:

- Headline model and training metrics loaded from the checkpoint and training report.
- The complete 21-point validation-loss curve.
- A measured MPS KV-cache benchmark with its methodology and caveats.
- Architecture, training provenance, intended use, limitations, and fixed samples.
- A streaming playground with length, temperature, top-k, and deterministic seed controls.

The interface does not create a public share link. Use `--no-browser` if you only want the local server, or `--device cpu` for CPU inference. The full model card is available in `MODEL_CARD.md`, while the machine-readable inference result is in `artifacts/inference_benchmark.json`.

### Frontend development

The editable shadcn/ui source is in `frontend/`. Vite proxies `/api` calls to the local Python server and writes production assets into `cleo1/static/`:

```bash
cd frontend
npm install
npm run dev       # HMR development server
npm run build     # production assets for `cleo-1 web`
```

The API exposes `/api/health`, `/api/profile`, canonical checkpoint identity at `/api/identity`, and a cancellable NDJSON generation stream at `/api/generate`. Direct identity questions use the verified identity record rather than probabilistic sampling.

## Tests

```bash
uv run pytest
cd frontend && npm run build && npm run lint
```

The suite covers tokenizer round trips and determinism, causal masking, parameter count, forward/backward behavior, cached/full-forward parity, deterministic CPU sampling, exact checkpoint continuation, identity metadata and prompt masking, tiny-corpus overfitting, token-file batching, FastAPI streaming and identity endpoints, packaged frontend assets, and an MPS smoke test when MPS is available.

## Limitations

This is a small educational story generator, not a chatbot or factual model. It has a short context window and narrow child-level training distribution. Generated text can be repetitive, inconsistent, biased, or inappropriate and must not be used for factual or safety-critical decisions.
