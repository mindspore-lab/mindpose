from typing import Any, Dict, Optional

from ...models.decoders import Decoder

from ...models.networks import EvalNet
from .inferencer import Inferencer
from .topdown_inferencer import TopDownHeatMapInferencer


def create_inferencer(
    net: EvalNet,
    name: str = "topdown_heatmap",
    config: Optional[Dict[str, Any]] = None,
    decoder: Optional[Decoder] = None,
    progress_bar: bool = False,
) -> Inferencer:
    if name == "topdown_heatmap":
        return TopDownHeatMapInferencer(
            net, config=config, decoder=decoder, progress_bar=progress_bar
        )
    else:
        raise ValueError(f"Unsupported inferencer: {name}")
