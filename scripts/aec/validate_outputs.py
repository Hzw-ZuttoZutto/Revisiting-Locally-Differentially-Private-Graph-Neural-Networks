from __future__ import annotations

import argparse
import json
from .paths import normalize_mode
from .search_results import TARGETS, validate_search_output


def validate_outputs(mode: str = "scaled") -> list[dict]:
    mode = normalize_mode(mode)
    if mode not in {"scaled", "full"}:
        raise ValueError("Search validation requires scaled or full mode")
    cache = {}
    return [validate_search_output(target, mode, cache=cache) for target in TARGETS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate search results, seed coverage and figure/table statistics.")
    parser.add_argument("--mode", choices=("scaled", "full"), default="scaled")
    args = parser.parse_args()
    print(json.dumps(validate_outputs(args.mode), indent=2))


if __name__ == "__main__":
    main()
