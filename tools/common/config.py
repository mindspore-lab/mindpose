import argparse
from typing import Any, Dict

import yaml


def create_parser(
    description: str = "", need_ckpt: bool = False
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path of the config file.")
    parser.add_argument(
        "--outdir", default="output", help="Path of the output directory"
    )
    parser.add_argument(
        "--ckpt", required=need_ckpt, help="Path of the trained checkpoint"
    )
    return parser


def parse_args(description: str = "", need_ckpt: bool = False) -> argparse.Namespace:
    parser = create_parser(description=description, need_ckpt=need_ckpt)
    args = parser.parse_args()
    cfg = parse_yaml(args.config)
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


def parse_yaml(fpath: str) -> Dict[str, Any]:
    with open(fpath) as f:
        cfg = yaml.safe_load(f)
    return cfg
