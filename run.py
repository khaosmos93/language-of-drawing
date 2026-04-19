"""CLI entry point.

    python run.py --input data/demo.png --out out/
    python run.py --demo                    # generate demo first, then run
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.demo_image import generate_demo
from core.pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Language of Drawing — image → 9 reps → reconstructions")
    p.add_argument("--input", type=str, default="data/demo.png")
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--demo", action="store_true", help="generate data/demo.png first")
    p.add_argument("--strategies", type=str, default="linear,nonlinear,pde")
    p.add_argument("--symbolic-degree", type=int, default=8)
    args = p.parse_args()

    if args.demo or not Path(args.input).exists():
        generate_demo(args.input)
        print(f"[demo] wrote {args.input}")

    arts = run_pipeline(
        args.input,
        args.out,
        strategies=tuple(s.strip() for s in args.strategies.split(",") if s.strip()),
        symbolic_degree=args.symbolic_degree,
    )
    print("[done] artifacts:")
    for k, v in arts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
