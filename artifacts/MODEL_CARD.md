# Cleo 1 — Model Card

## Model summary

Cleo 1 (`cleo-1`) is a 7,809,024-parameter decoder-only transformer developed and trained by Cleo AI from random initialization for short, child-level fictional story continuation. The model, tokenizer, attention implementation, training loop, checkpoint format, and sampling code were built locally without pretrained weights or a pretrained tokenizer.

This is a narrow educational language model, not a general assistant.

## Model details

| Field | Value |
| --- | --- |
| Model name | Cleo 1 |
| Model ID | `cleo-1` |
| Company | Cleo AI |
| Release | Research release 01 |
| Release checkpoint | `artifacts/cleo-1.pt` |
| Story-quality base checkpoint | `artifacts/best.pt` |
| Parameters | 7,809,024 |
| Architecture | Decoder-only, pre-normalization transformer |
| Layers | 6 |
| Embedding width | 320 |
| Attention heads | 5 × 64 dimensions |
| Feed-forward width | 1,280 |
| Context | 256 tokens |
| Tokenizer | Custom lossless byte-level BPE |
| Vocabulary | 1,024 tokens |
| Precision | FP32 |
| Training device | Apple M4 through PyTorch MPS |
| Foundation training steps | 20,000 |
| Identity fine-tuning steps | 300 |
| Training duration | 9,333 seconds (approximately 2h 35m) |

Input and output embeddings are tied. Each block uses explicit causal self-attention, GELU feed-forward layers, dropout 0.1, and pre-LayerNorm. Generation uses temperature/top-k sampling and an exact per-layer key/value cache that is rebuilt when the learned-position context window rolls over.

## Training data

The training corpus is the first official Parquet training shard of `roneneldan/TinyStories`, with the complete validation split.

- Dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Dataset license: CDLA-Sharing-1.0
- Training stories: 529,875
- Prepared training tokens: 234,379,409
- Validation stories: 21,990
- Prepared validation tokens: 9,412,338
- Training-token presentations: 163,840,000
- Tokenizer training sample: 8 MiB from the pinned training corpus

Source URLs, byte sizes, and SHA-256 checksums are recorded in `data/processed/manifest.json`. The tokenizer artifact records its merge order, special-token IDs, source revision, corpus checksum, and its own checksum.

## Training procedure

The model was optimized with AdamW using a peak learning rate of `6e-4`, betas `(0.9, 0.95)`, weight decay `0.1`, gradient clipping at `1.0`, and an effective batch of 8,192 tokens. A 500-step warmup was followed by cosine decay to `6e-5`. Validation used 50 fixed batches every 1,000 steps.

Checkpoints include model and optimizer state, configuration, RNG states, tokenizer checksum, dataset manifest, step, elapsed training time, and validation history.

### Identity fine-tuning

The release checkpoint was continued from `artifacts/best.pt` for 300 supervised steps at a learning rate of `2e-5`. Each update combined varied identity-answer examples with the original story corpus at a 4:1 story-to-identity loss weight. The canonical target is: “I am Cleo 1. My model ID is cleo-1. I was developed and trained by Cleo AI.”

The base checkpoint is preserved separately. The release was accepted only after exact greedy generation on every held-out identity paraphrase, a paired story-validation loss increase below 3%, and zero identity leakage in full-length seeded story probes.

## Evaluation results

| Metric | Result |
| --- | ---: |
| Initial validation cross-entropy | 6.8710 |
| Best validation cross-entropy | 1.0661 |
| Relative validation-loss reduction | 84.48% |
| Best validation perplexity | 2.9041 |
| Release validation cross-entropy | 1.0668 |
| Release validation perplexity | 2.9062 |
| Release/base loss change on standard evaluation | +0.07% |
| Acceptance gate | Passed |
| Final step | 20,000 |
| Identity held-out exact match | 8/8 (100%) |
| Identity fine-tuning steps | 300 |
| Paired story loss before identity tune | 1.0528 |
| Paired story loss after identity tune | 1.0542 |
| Paired story loss change | +0.13% |
| Identity leakage in fixed story probes | 0/5 |

These validation metrics measure next-token prediction on the pinned TinyStories validation token stream. They are tokenizer- and dataset-specific, do not establish broad language understanding, and should not be compared directly with differently tokenized or differently evaluated models.

Identity exact match measures a narrow, deliberately memorized self-identification behavior on eight held-out paraphrases. It does not imply self-awareness, general instruction following, or broad factual reliability. The API also serves the canonical checkpoint identity deterministically for direct identity questions.

### Local inference benchmark

A warmed-up Apple M4 MPS run generated 128 new tokens in FP32 with batch size 1 and top-k 1.

| Path | Throughput | Time |
| --- | ---: | ---: |
| KV cached | 86.50 tokens/s | 1.480 s |
| Full forward | 39.87 tokens/s | 3.211 s |
| Speedup | 2.17× | — |

The outputs were token-identical for this benchmark. This is a single local engineering benchmark, not a standardized cross-model leaderboard result. Performance varies with prompt length, thermals, and system load. The full record is in `artifacts/inference_benchmark.json`.

## Intended use

Cleo 1 is intended for:

- Educational study of small language models.
- Short, child-level fictional story continuation.
- Inspecting tokenization, causal attention, training, checkpointing, and inference.
- Local, offline demonstrations on compatible Apple Silicon or CPU systems.
- Controlled experiments where model size and provenance matter more than broad capability.

## Out-of-scope use

Do not use Cleo 1 as:

- A chatbot or general-purpose assistant.
- A factual, medical, legal, financial, or safety-critical information source.
- A reasoning, planning, or decision-making system.
- An educational assessment or child-safety system.
- A substitute for human review in any consequential workflow.

## Limitations and risks

The model can repeat phrases, contradict itself, lose track of characters, end abruptly, or generate unsuitable content. It has a short 256-token attention window and a narrow training distribution. Byte-level tokenization supports arbitrary UTF-8 input without unknown tokens, but the training data is primarily simple English and does not imply multilingual capability.

No dedicated safety alignment, adversarial evaluation, toxicity filtering, preference optimization, or human-feedback training was performed. TinyStories and generated outputs may reflect biases or undesirable patterns from source data. Fixed samples are deterministic and unedited so that both strengths and failure modes remain visible.

## Reproducibility and provenance

- Random seed: 1337
- Source revision and checksums: `data/processed/manifest.json`
- Ordered tokenizer merges: `data/processed/tokenizer.json`
- Training history: `artifacts/metrics.jsonl`
- Training report: `artifacts/training_report.json`
- Identity fine-tuning report: `artifacts/identity_finetune_report.json`
- Identity fine-tuning metrics: `artifacts/identity_metrics.jsonl`
- Release validation check: `artifacts/release_evaluation.json`
- Fixed generations: `artifacts/samples.json`
- Inference benchmark: `artifacts/inference_benchmark.json`
- Architecture and training defaults: `configs/tinystories_m4.toml`

The checkpoint and tokenizer checksum are verified together when the model is loaded.

## Licenses

Project code is provided under the repository’s MIT license. TinyStories is distributed under CDLA-Sharing-1.0; users should review that license and the dataset terms when redistributing derived artifacts.
