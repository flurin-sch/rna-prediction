"""Supplied chromosome split and both cell-transfer evaluation designs."""
EXPERIMENTS = {
    "a_to_b": {"train_cells": ("X1", "X2"), "val_cells": ("X1", "X2"),
               "train_splits": ("train",), "val_splits": ("val",)},
    "x1_to_x2": {"train_cells": ("X1",), "val_cells": ("X2",),
                 "train_splits": ("train", "val"), "val_splits": ("train", "val")},
    "x2_to_x1": {"train_cells": ("X2",), "val_cells": ("X1",),
                 "train_splits": ("train", "val"), "val_splits": ("train", "val")},
}
LABELS = {"a_to_b": "A → B", "x1_to_x2": "X1 → X2", "x2_to_x1": "X2 → X1"}

