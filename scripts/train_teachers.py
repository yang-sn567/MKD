from __future__ import annotations

import argparse

from mkd_mpp.config import load_config
from mkd_mpp.training import TeacherTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unimodal MKD teachers.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--one-d-path", default=None)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--geometry-path", default=None)
    parser.add_argument("--llm-text-path", default=None)
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_path is not None:
        config.data.data_path = args.data_path
    if args.one_d_path is not None:
        config.data.one_d_path = args.one_d_path
    if args.graph_path is not None:
        config.data.graph_path = args.graph_path
    if args.geometry_path is not None:
        config.data.geometry_path = args.geometry_path
    if args.llm_text_path is not None:
        config.data.llm_text_path = args.llm_text_path
    if args.task_type is not None:
        config.data.task_type = args.task_type
    TeacherTrainer(config).train_all()


if __name__ == "__main__":
    main()
