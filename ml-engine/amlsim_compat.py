"""Isolated Python 3.12 compatibility runner for IBM AMLSim's legacy generator.

Install NetworkX 1.11/decorator 4.4.2 into --deps, never into the ML environment.
No transaction generation rules or labels are modified by this runner.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import fractions
import json
import math
from pathlib import Path
import random
import runpy
import sys
from typing import Any


def main() -> None:
    """Apply API/newline compatibility shims in a separate Python process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    fractions.gcd = math.gcd  # type: ignore[attr-defined]
    sys.path.insert(0, str(args.deps.resolve()))
    sys.path.insert(0, str(args.repo.resolve() / "scripts"))
    original = random.sample

    def compatible_sample(population: Any, k: int, *, counts: Any = None) -> list[Any]:
        """Support legacy set sampling while making iteration deterministic."""
        if not isinstance(population, Sequence):
            population = sorted(population)
        return original(population, k, counts=counts)

    random.sample = compatible_sample
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sys.argv = [str(args.repo / "scripts/transaction_graph_generator.py"), str(args.config)]
    runpy.run_path(sys.argv[0], run_name="__main__")
    folder = Path(config["temporal"]["directory"]) / config["general"]["simulation_name"]
    for path in folder.glob("*.csv"):
        path.write_bytes(path.read_bytes().replace(b"\r\r\n", b"\r\n"))


if __name__ == "__main__":
    main()
