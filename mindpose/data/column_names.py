# Columns used for dataset and pipeline

# TOPDOWN
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

# BOTTOMUP

_BOTTOMUP_TRAIN_COLUMN_NAMES = [
    "image",
    "boxes",
    "keypoints",
    "target",
    "mask",
    "tag_ind",
]

_BOTTOMUP_TRAIN_FINAL_COLUMN_NAMES = ["image", "target", "mask", "tag_ind"]

_BOTTOMUP_VAL_COLUMN_NAMES = [
    "image",
    "mask",
    "center",
    "scale",
    "image_file",
    "image_shape",
]

_BOTTOMUP_VAL_FINAL_COLUMN_NAMES = [
    "image",
    "mask",
    "center",
    "scale",
    "image_file",
    "image_shape",
]

# TODO: Looks ugly, need refactor
COLUMN_MAP = dict(
    coco_topdown=dict(train=_TOPDOWN_TRAIN_COLUMN_NAMES, val=_TOPDOWN_VAL_COLUMN_NAMES),
    topdown=dict(train=_TOPDOWN_TRAIN_COLUMN_NAMES, val=_TOPDOWN_VAL_COLUMN_NAMES),
    coco_bottomup=dict(
        train=_BOTTOMUP_TRAIN_COLUMN_NAMES, val=_BOTTOMUP_VAL_COLUMN_NAMES
    ),
    bottomup=dict(train=_BOTTOMUP_TRAIN_COLUMN_NAMES, val=_BOTTOMUP_VAL_COLUMN_NAMES),
    imagefolder_bottomup=dict(val=_BOTTOMUP_VAL_COLUMN_NAMES),
)

FINAL_COLUMN_MAP = dict(
    topdown=dict(
        train=_TOPDOWN_TRAIN_FINAL_COLUMN_NAMES, val=_TOPDOWN_VAL_FINAL_COLUMN_NAMES
    ),
    bottomup=dict(
        train=_BOTTOMUP_TRAIN_FINAL_COLUMN_NAMES, val=_BOTTOMUP_VAL_FINAL_COLUMN_NAMES
    ),
    imagefolder_bottomup=dict(val=_BOTTOMUP_VAL_FINAL_COLUMN_NAMES),
)
