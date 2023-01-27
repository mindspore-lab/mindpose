from typing import Any, Dict, List, Optional

from mindspore.dataset import Dataset

from ...models.networks import EvalNet


class Inferencer:
    def __init__(self, net: EvalNet, config: Optional[Dict[str, Any]] = None) -> None:
        self.net = net
        net.set_train(False)
        self.config = config if config else dict()
        self._inference_cfg = self.load_inference_cfg()

    def load_inference_cfg(self) -> Dict[str, Any]:
        raise NotImplementedError("Child Class must implement this method.")

    def __call__(self, dataset: Dataset) -> List[Dict[str, Any]]:
        raise NotImplementedError("Child class must implement this method.")
