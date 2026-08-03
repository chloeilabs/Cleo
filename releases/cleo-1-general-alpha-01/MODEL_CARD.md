# Cleo 1 — Model Card

## Model summary

Cleo 1 (`cleo-1`) is a 7,890,944-parameter decoder-only transformer developed and trained by Cleo AI from random initialization. General-language alpha 01 extends the original language foundation with continued pretraining on WikiText-103, answer-only instruction tuning on Databricks Dolly 15K, a context expansion from 256 to 512 tokens, and a bounded identity repair.

No pretrained weights or pretrained tokenizer were used. This is an experimental small-model research release, not a reliable general-purpose assistant.

## Model details

| Field | Value |
| --- | --- |
| Model name | Cleo 1 |
| Model ID | `cleo-1` |
| Developer and trainer | Cleo AI |
| Release | `cleo-1-general-alpha-01` (frozen) |
| Release checkpoint | `artifacts/cleo-1.pt` / GitHub Release `cleo-1-general-alpha-01` |
| Successor | Cleo 1.1 (~135M; see `configs/cleo11_135m.toml`) |
| Preserved pre-generalization checkpoint | `artifacts/cleo-1-story.pt` |
| Parameters | 7,890,944 |
| Architecture | Decoder-only, pre-normalization transformer |
| Layers | 6 |
| Embedding width | 320 |
| Attention heads | 5 × 64 dimensions |
| Feed-forward width | 1,280 |
| Context | 512 tokens |
| Tokenizer | Custom lossless byte-level BPE |
| Vocabulary | 1,024 tokens |
| Precision | FP32 |
| Training device | Apple M4 through PyTorch MPS |
| Final step | 23,000 |
| Total recorded training time | 11,333 seconds (about 3h 9m) |

Input and output embeddings are tied. Each block uses explicit causal self-attention, GELU feed-forward layers, dropout 0.1, and pre-LayerNorm. Positions are learned. The context expansion preserves the original 256 position embeddings and randomly initializes the additional positions. Generation supports temperature/top-k sampling and a per-layer key/value cache.

## Training data

### Foundation corpus

- Dataset: `roneneldan/TinyStories`
- Revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- License: CDLA-Sharing-1.0
- Train: 529,875 examples and 234,379,409 prepared tokens
- Validation: 21,990 examples and 9,412,338 prepared tokens

### General-language corpus

- Dataset: `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`
- Revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- License: CC BY-SA 3.0 and GFDL
- Train: 1,165,029 nonempty documents and 335,385,799 prepared tokens
- Validation: 2,461 nonempty documents and 713,052 prepared tokens
- Test: 2,891 nonempty documents and 802,787 prepared tokens

### Instruction corpus

- Dataset: `databricks/databricks-dolly-15k`
- Revision: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`
- License: CC BY-SA 3.0
- Deterministic category-stratified split: 13,511 train, 750 validation, 750 test
- Categories: brainstorming, classification, closed QA, creative writing, general QA, information extraction, open QA, and summarization

All sources are pinned by revision, expected byte size, and SHA-256. Processed files and the tokenizer are also checksummed. Details are in `data/processed/manifest.json`, `data/general/processed/manifest.json`, and `configs/general_m4.toml`.

## Training procedure

1. Train the 7.81M-parameter, 256-context foundation from random weights for 20,000 AdamW steps.
2. Run 300 supervised identity steps, with foundation-data retention, to establish the canonical Cleo AI identity.
3. Expand context to 512 positions and continue pretraining for 2,000 steps on WikiText, with a 10% probability of a foundation-data microbatch.
4. Run 600 answer-only instruction-tuning steps on Dolly. Each step also includes weighted general-language, foundation, and identity-retention losses.
5. Run a low-learning-rate identity repair. The accepted run stopped at 100 steps after every release gate passed.

The general continued-pretraining stage used AdamW, a peak learning rate of `2e-4`, 100-step warmup, cosine decay to `2e-5`, weight decay `0.1`, gradient clipping at `1.0`, and an effective 8,192 autoregressive tokens per optimizer step. Instruction tuning used `5e-5`; identity repair used `1e-5`. Seed 1337 was used throughout.

Checkpoints include weights, configuration, optimizer and scheduler state where applicable, RNG states, tokenizer checksum, source manifests, stage, step, elapsed time, and acceptance metrics.

## Evaluation results

| Metric | Before generalization | Released alpha | Change |
| --- | ---: | ---: | ---: |
| WikiText validation cross-entropy | 4.1848 | 2.0439 | −51.2% |
| WikiText validation perplexity | 65.68 | 7.7204 | −88.2% |
| Dolly answer-token validation cross-entropy | 4.1376 | 1.8441 | −55.4% |
| Foundation validation cross-entropy | 1.0602 | 1.2401 | +17.0% |
| Held-out identity exact match | 8/8 before generalization | 8/8 after repair | retained |

The promotion thresholds were a general-loss ratio at or below 0.80, an instruction-loss ratio at or below 0.65, a foundation-loss ratio at or below 1.35, and identity exact match of 100%. The released checkpoint passed all four. The identity repair itself changed general, instruction, and foundation losses by less than 0.5% relative to the unrepaired general candidate.

These losses measure token prediction on specific pinned distributions with Cleo 1's tokenizer. They are not standardized cross-model benchmarks and do not prove reasoning, factuality, usefulness, or safety. Models with different tokenizers or data should not be compared directly using these values.

### Qualitative and capability evaluation

Five fixed prompts expose the release's open-ended behavior in `artifacts/general_samples.json`. Responses remain repetitive and often incorrect. A separate synthetic curriculum covering arithmetic, comparison, sentiment, uppercase transformation, and extraction improved some narrow probes but failed the declared release gates and degraded open-ended output. Its candidate was rejected and not promoted.

Identity exact match is a deliberately trained self-identification behavior on eight held-out paraphrases. It does not imply self-awareness. The API also returns the canonical checkpoint identity deterministically for direct identity questions.

### Local inference benchmark

On the promoted checkpoint, a warmed Apple M4 MPS benchmark generated 128 tokens at a five-run median of 142.61 tokens/s with the KV cache and 120.29 tokens/s with full-forward decoding, a 1.19× speedup. Settings were FP32, batch size 1, 47 prompt tokens, temperature 1.0, and top-k 1. Outputs were token-identical. This is a local engineering measurement, not a cross-model benchmark; thermals and system load affect results. The full run timings are in `artifacts/general_inference_benchmark.json`.

## Intended use

Cleo 1 is intended for:

- Educational study of small language models trained without pretrained components.
- Controlled local experiments in language modeling and instruction formatting.
- Inspecting tokenization, causal attention, context expansion, training, gating, checkpointing, and inference.
- Reproducible demonstrations on compatible Apple Silicon or CPU systems.
- Research where transparency and small scale matter more than answer quality.

## Out-of-scope use

Do not use Cleo 1 as:

- A production chatbot or dependable general-purpose assistant.
- A factual, medical, legal, financial, or safety-critical information source.
- A reasoning, planning, coding, or autonomous decision-making system.
- An educational assessment, moderation, or child-safety system.
- A substitute for qualified human review in any consequential workflow.

## Limitations and risks

At 7.89M parameters, the model has very limited capacity and knowledge. It frequently repeats phrases, gives incorrect answers, ignores constraints, contradicts itself, or generates unsuitable text. Arithmetic, coding, multi-step reasoning, long-context recall, factual calibration, multilingual behavior, and robust instruction following are not established capabilities. The 512-token context remains short.

No dedicated safety alignment, adversarial safety evaluation, toxicity filtering, preference optimization, or human-feedback training was performed. Source data and generated outputs may contain biases or undesirable material. Byte-level tokenization can represent arbitrary UTF-8 without unknown tokens, but representability is not multilingual competence.

## Reproducibility and provenance

- Seed: `1337`
- Architecture and foundation defaults: `configs/tinystories_m4.toml`
- Generalization data, schedule, mixing, and gates: `configs/general_m4.toml`
- Foundation data manifest: `data/processed/manifest.json`
- General and instruction data manifest: `data/general/processed/manifest.json`
- Ordered tokenizer merges: `data/processed/tokenizer.json`
- Foundation training history: `artifacts/metrics.jsonl`
- Generalization history: `artifacts/generalization_metrics.jsonl`
- Generalization report: `artifacts/generalization_report.json`
- Identity-repair history: `artifacts/general_identity_repair_metrics.jsonl`
- Accepted repair report: `artifacts/general_identity_repair_report.json`
- Machine-readable release evaluation: `artifacts/general_release_evaluation.json`
- Fixed qualitative probes: `artifacts/general_samples.json`
- Current-checkpoint inference benchmark: `artifacts/general_inference_benchmark.json`

The promoted checkpoint SHA-256 is `9bfb8544b39cdd60f6b57bbcf85dd3b5fc87b15084535fcbfa75d97e3130390f`. The checkpoint and tokenizer checksum are verified together at load time.

## Freeze status

This alpha is frozen. Do not expect additional Cleo 1 fine-tunes to produce a generally capable assistant. Packaging metadata for the frozen tag lives in `releases/cleo-1-general-alpha-01/`. Cleo 1.1 is the capacity-expanding successor: ~135M parameters, 2,048-token context, 16K byte-level BPE, RoPE/RMSNorm/SwiGLU/GQA, FineWeb-Edu-led pretraining, and capability-gated promotion.

## Licenses

Project code is MIT licensed. TinyStories is distributed under CDLA-Sharing-1.0. WikiText is distributed under CC BY-SA 3.0 and GFDL. Databricks Dolly 15K is distributed under CC BY-SA 3.0. Review each dataset's terms before redistributing data or derived artifacts.
