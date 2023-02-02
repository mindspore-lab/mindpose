from typing import Any, Dict, List, Optional

from mindspore.dataset import Dataset

from ...models import EvalNet


class Inferencer:
    """Create an inference engine
    This is an abstract class, child class must implement `load_inference_cfg` and `__call__` method.

    Args:
        net: Network for evaluation
        config: Method-specific configuration. Default: None

    Inputs:
        dataset: Dataset

    Outputs:
        records: List of inference records
    """

    def __init__(self, net: EvalNet, config: Optional[Dict[str, Any]] = None) -> None:
        self.net = net
        net.set_train(False)
        self.config = config if config else dict()
        self._inference_cfg = self.load_inference_cfg()

    def load_inference_cfg(self) -> Dict[str, Any]:
        """Loading the inference config, where the returned config must be a dictionary
        which stores the configuration of the engine, such as the using TTA, etc.
        """
        raise NotImplementedError("Child Class must implement this method.")

    def __call__(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Running the inference on the dataset. And return a list of records.
        Normally, in order to be compatible with the evaluator engine,
        each record should contains the following keys:

        Keys:
            pred: The predicted coordindate, in shape [M, (x_coord, y_coord, score)]
            box: The coor bounding boxes, in shape [(center_x, center_y, scale_x, scale_y, area, bounding box score)]
            image_path: The path of the image
            bbox_id: Bounding box ID
        """
        raise NotImplementedError("Child class must implement this method.")
