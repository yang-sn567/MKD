from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mkd_mpp.config import load_config
from mkd_mpp.training import MKDTrainer, TeacherTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MKD-MPP teacher and student training.")
    parser.add_argument("--config", default="configs/pcqm4mv2_100k.yaml")
    parser.add_argument("--one-d-path", default=None)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--geometry-path", default=None)
    parser.add_argument("--llm-text-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.one_d_path is not None:
        config.data.one_d_path = args.one_d_path
        config.data.data_path = args.one_d_path
    if args.graph_path is not None:
        config.data.graph_path = args.graph_path
    if args.geometry_path is not None:
        config.data.geometry_path = args.geometry_path
    if args.llm_text_path is not None:
        config.data.llm_text_path = args.llm_text_path

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    results = {}
    if config.training.train_teachers:
        results["teachers"] = TeacherTrainer(config).train_all()
    if config.training.train_student:
        results["student"] = MKDTrainer(config).train()

    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
