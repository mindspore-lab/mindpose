#!/usr/bin/env python
"""Runing training on OpenI
"""
import argparse
import functools
import os
import subprocess
import sys
import time
from typing import Any, Callable

import moxing as mox
from common.config import create_parser, parse_yaml


LOCAL_RANK = int(os.getenv("RANK_ID", 0))


def run_with_single_rank(
    local_rank: int = 0, signal: str = "/tmp/SUCCESS"
) -> Callable[..., Any]:
    """Run the task on 0th rank, perform synchronzation before return"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if local_rank == 0:
                result = func(*args, **kwargs)
                with open(signal, "w") as f:
                    f.write("\n")
                return result
            else:
                while not os.path.isfile(signal):
                    time.sleep(1)

        return wrapper

    return decorator


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/INSTALL_SUCCESS")
def install_packages(project_dir: str) -> None:
    url = "https://pypi.tuna.tsinghua.edu.cn/simple"
    requirement_txt = os.path.join(project_dir, "requirements.txt")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-i", url, "-r", requirement_txt]
    )


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/DOWNLOAD_DATA_SUCCESS")
def download_data(s3_path: str, dest: str) -> None:
    if not os.path.isdir(dest):
        os.makedirs(dest)

    mox.file.copy_parallel(src_url=s3_path, dst_url=dest)


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/DOWNLOAD_CKPT_SUCCESS")
def download_ckpt(s3_path: str, dest: str) -> str:
    if not os.path.isdir(dest):
        os.makedirs(dest)

    filename = os.path.basename(s3_path)
    dst_url = os.path.join(dest, filename)

    mox.file.copy(src_url=s3_path, dst_url=dst_url)
    return dst_url


def upload_data(src: str, s3_path: str) -> None:
    abs_src = os.path.abspath(src)
    print(f"Uploading data from {abs_src} to s3")
    mox.file.copy_parallel(src_url=abs_src, dst_url=s3_path)


def parse_args(
    project_dir: str,
    description: str = "",
    need_ckpt: bool = False,
) -> argparse.Namespace:
    parser = create_parser(description=description, need_ckpt=need_ckpt)
    # add arguments
    # the follow arguments are proviced by OpenI, do not change.
    parser.add_argument("--device_target", help="Device target")
    parser.add_argument("--data_url", help="Path of the data url in S3")
    parser.add_argument("--train_url", help="Path of the training output in S3")
    parser.add_argument("--ckpt_url", help="Path of the ckpt in S3")
    args = parser.parse_args()

    config_path = os.path.join(project_dir, args.config)
    cfg = parse_yaml(config_path)
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    print(os.environ)

    # locate the path of the project
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # install necessary packages
    install_packages(project_dir)

    from train import train

    args = parse_args(project_dir, description="Training script on OpenI")

    # copy data from S3 to local
    local_data_path = "/home/work/data"
    download_data(s3_path=args.data_url, dest=local_data_path)

    # symlink the data directory to path in configure file
    os.symlink(local_data_path, "data", target_is_directory=True)

    # copy ckpt from S3 to local
    if args.ckpt_url:
        args.ckpt = download_ckpt(s3_path=args.ckpt_url, dest="/tmp/ckpt/")

    # start traning job
    train(args)

    # copy output from local to s3
    time.sleep(30)
    if LOCAL_RANK == 0:
        upload_data(src=args.outdir, s3_path=args.train_url)
