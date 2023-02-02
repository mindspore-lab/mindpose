from typing import Any, Dict, List, Optional

from mindspore.dataset import Dataset

from ...models import EvalNet


class Inferencer:
    """Create an inference engine. It runs the inference on the entire dataset and
    outputs a list of records.

    Args:
        net: Network for inference
        config: Method-specific configuration for inference. Default: None

    Inputs:
        | dataset: Dataset for inferencing

    Outputs:
        | records: List of inference records

    Note:
        This is an abstract class, child class must implement
        `load_inference_cfg` method.
    """

    def __init__(self, net: EvalNet, config: Optional[Dict[str, Any]] = None) -> None:
        self.net = net
        net.set_train(False)
        self.config = config if config else dict()
        self._inference_cfg = self.load_inference_cfg()

    def load_inference_cfg(self) -> Dict[str, Any]:
        """Loading the inference config, where the returned config must be a dictionary
        which stores the configuration of the engine, such as the using TTA, etc.

        Returns:
            Inference configurations
        """
        raise NotImplementedError("Child Class must implement this method.")

    def infer(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Running the inference on the dataset. And return a list of records.
        Normally, in order to be compatible with the evaluator engine,
        each record should contains the following keys:

        Keys:
            | pred: The predicted coordindate, in shape [C, 3(x_coord, y_coord, score)]
            | box: The coor bounding boxes, each record contains
                (center_x, center_y, scale_x, scale_y, area, bounding box score)
            | image_path: The path of the image
            | bbox_id: Bounding box ID

        Args:
            dataset: Dataset for inferencing

        Returns:
            List of inference results
        """
        raise NotImplementedError("Child class must implement this method.")

    def __call__(self, dataset: Dataset) -> List[Dict[str, Any]]:
        return self.infer(dataset)
