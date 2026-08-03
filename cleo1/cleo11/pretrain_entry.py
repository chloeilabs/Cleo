"""torchrun entry module: ``torchrun -m cleo1.cleo11.pretrain_entry``."""

from __future__ import annotations

import argparse
import json
import sys

from .pretrain import pretrain_from_config_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cleo 1.1 distributed pretrain entrypoint")
    parser.add_argument("--cleo11-config", default="configs/cleo11_135m.toml")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--microbatch-size", type=int)
    args = parser.parse_args(argv)
    result = pretrain_from_config_path(
        args.cleo11_config,
        requested_device=args.device,
        resume_path=args.resume,
        max_steps_override=args.max_steps,
        microbatch_override=args.microbatch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
