import argparse
import logging
from ast import literal_eval
from typing import Any, Dict

import yaml

_logger = logging.getLogger(__name__)


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            if "=" in kv:
                k, v = kv.split("=", maxsplit=1)
            elif ":" in kv:
                k, v = kv.split(":", maxsplit=1)
            else:
                raise ValueError(
                    "it must be `KEY1=VAL1, ...` or `KEY1:VAL1, ...` pairs."
                )

            try:
                my_dict[k] = literal_eval(v)
            except Exception:
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


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
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=StoreDictKeyPair,
        default=dict(),
        metavar="KEY1=VAL1 KEY2=VAL2 ...",
    )
    return parser


def parse_args(description: str = "", need_ckpt: bool = False) -> argparse.Namespace:
    parser = create_parser(description=description, need_ckpt=need_ckpt)
    args = parser.parse_args()

    # read the config file and overwrite them into the Namespace
    merge_cfg_from_yaml(args, args.config)

    # read the config from cfg-options and over write the Namespace
    merge_cfg_options(args)

    _logger.info(args)

    return args


def merge_cfg_from_yaml(args: argparse.Namespace, config_path: str):
    cfg = parse_yaml(config_path)
    for k, v in cfg.items():
        setattr(args, k, v)


def merge_cfg_options(args: argparse.Namespace) -> None:
    for k, v in args.cfg_options.items():
        key_list = k.split(".")
        subargs = args
        for subkey in key_list[:-1]:
            try:
                subargs = getattr(subargs, subkey)
            except AttributeError:
                subargs = subargs[subkey]
        subkey = key_list[-1]
        try:
            setattr(subargs, subkey, v)
        except AttributeError:
            subargs[subkey] = v

    del args.cfg_options


def parse_yaml(fpath: str) -> Dict[str, Any]:
    with open(fpath) as f:
        cfg = yaml.safe_load(f)
    return cfg
