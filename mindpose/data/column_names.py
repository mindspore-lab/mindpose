# Columns used for dataset and pipeline
_TOPDOWN_TRAIN_COLUMN_NAMES = [
    "image",
    "center",
    "scale",
    "boxes",
    "keypoints",
    "rotation",
    "target",
    "target_weight",
]

_TOPDOWN_TRAIN_FINAL_COLUMN_NAMES = ["image", "target", "target_weight"]

_TOPDOWN_VAL_COLUMN_NAMES = [
    "image",
    "center",
    "scale",
    "rotation",
    "image_file",
    "boxes",
    "bbox_ids",
    "bbox_scores",
]

_TOPDOWN_VAL_FINAL_COLUMN_NAMES = [
    "image",
    "image_file",
    "boxes",
    "bbox_ids",
    "center",
    "scale",
    "bbox_scores",
]


COLUMN_MAP = dict(
    coco_topdown=dict(train=_TOPDOWN_TRAIN_COLUMN_NAMES, val=_TOPDOWN_VAL_COLUMN_NAMES),
    topdown=dict(train=_TOPDOWN_TRAIN_COLUMN_NAMES, val=_TOPDOWN_VAL_COLUMN_NAMES),
)

FINAL_COLUMN_MAP = dict(
    topdown=dict(
        train=_TOPDOWN_TRAIN_FINAL_COLUMN_NAMES, val=_TOPDOWN_VAL_FINAL_COLUMN_NAMES
    )
)
