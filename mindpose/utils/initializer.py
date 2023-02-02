""" network initialized related"""
import logging
import math
from functools import reduce
from typing import Tuple

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import initializer as init


def _calculate_fan_in_and_fan_out(arr: Tensor) -> Tuple[int, int]:
    """Calculate fan in and fan out."""
    dimensions = len(arr.shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for array with fewer than 2 dimensions"
        )

    num_input_fmaps = arr.shape[1]
    num_output_fmaps = arr.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = reduce(lambda x, y: x * y, arr.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def init_by_kaiming_uniform(network: nn.Cell) -> None:
    """Initialize the network parameters by Kaiming Uniform

    Args:
        network: Network to initialize
    """
    logging.info("Initialize the network parameters with Kaiming Uniform.")

    for name, cell in network.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            logging.debug(f"Reinitialize {name}")
            cell.weight.set_data(
                init.initializer(init.HeUniform(), cell.weight.shape, cell.weight.dtype)
            )
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(
                    init.initializer(
                        init.Uniform(bound), cell.bias.shape, cell.bias.dtype
                    )
                )
        elif isinstance(cell, nn.Dense):
            logging.debug(f"Reinitialize {name}")
            cell.weight.set_data(
                init.initializer(init.HeUniform(), cell.weight.shape, cell.weight.dtype)
            )
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(
                    init.initializer(
                        init.Uniform(bound), cell.bias.shape, cell.bias.dtype
                    )
                )
        else:
            logging.debug(f"Skip {name}")
