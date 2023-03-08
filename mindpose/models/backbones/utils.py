import logging
import os

import mindspore as ms

from ...utils.download import DownLoad
from .backbone import Backbone

_logger = logging.getLogger(__name__)


def load_pretrained(
    net: Backbone, ckpt_url: str, download_dir: str = "./cache/pretrained/"
) -> None:
    """Loading the pretrained backbone's weight from url.

    Args:
        net: network to load pretrained weight
        ckpt_url: If is a url, it is downloaded to local and then load.
            If it is a local path, it is loaded directly
        download_dir: The path storing the downloaed checkpoint
    """
    if os.path.isfile(ckpt_url):
        local_ckpt_path = ckpt_url
    else:
        # download files
        os.makedirs(download_dir, exist_ok=True)
        _logger.info(
            f"Saving the pretrained weight from `{ckpt_url}` to `{download_dir}`"
        )
        try:
            DownLoad().download_url(ckpt_url, path=download_dir)
        except Exception as e:
            raise ValueError(
                f"Cannot download the pretrained weight from `{ckpt_url}`."
            ) from e
        local_ckpt_path = os.path.join(download_dir, os.path.basename(ckpt_url))

    _logger.info(f"Loading the checkpoint from `{local_ckpt_path}`")

    try:
        ms.load_checkpoint(local_ckpt_path, net=net, strict_load=False)
    except ValueError as e:
        _logger.warning(f"Cannot load checkpoint from {local_ckpt_path}. ({str(e)})")
