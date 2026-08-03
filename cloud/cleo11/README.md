# Cleo 1.1 cloud pretrain

Portable CUDA Docker + `torchrun` launcher for the 135M Cleo 1.1 run.

## Mounts

- `data/cleo11` → prepared tokenizer, shards, manifest
- `artifacts/cleo11` → checkpoints and metrics

## Quick start on a GPU VM

```bash
# From the repository root
export HF_TOKEN=...   # required when CLEO11_PREPARE=1 streams Hugging Face data

uv run cleo-1 cleo11-launch --profile full --emit-script /tmp/cleo11-launch.sh
bash /tmp/cleo11-launch.sh
```

Or manually:

```bash
docker build -f cloud/cleo11/Dockerfile -t cleo11-pretrain:latest .
docker run --gpus all --rm \
  -v "$(pwd)/data/cleo11:/workspace/data/cleo11" \
  -v "$(pwd)/artifacts/cleo11:/workspace/artifacts/cleo11" \
  -e CLEO11_PROFILE=full \
  -e CLEO11_PREPARE=1 \
  -e HF_TOKEN \
  cleo11-pretrain:latest
```

Set `CLEO11_PREPARE=0` if shards were prepared ahead of time.

## Profiles

| Profile | Train tokens | Validation tokens | Intended host |
| --- | ---: | ---: | --- |
| `dev` | 50M | 2M | local prep / short cloud debug |
| `full` | 2.72B | 10M | cloud CUDA pretrain |

The M4 remains for development, evaluation, and inference — not the full 135M pretrain.
