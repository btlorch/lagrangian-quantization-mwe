import torch
import numpy as np
import eagerpy as ep
import foolbox


def onehot_like(self: torch.Tensor, indices: torch.Tensor, *, value: float = 1) -> torch.Tensor:
    if self.ndim != 2:
        raise ValueError("onehot_like only supported for 2D tensors")
    if indices.ndim != 1:
        raise ValueError("onehot_like requires 1D indices")
    if len(indices) != len(self):
        raise ValueError("length of indices must match length of tensor")

    x = torch.zeros_like(self)
    rows = np.arange(x.shape[0])
    x[rows, indices] = value
    return x


def best_other_classes(logits: torch.Tensor, exclude: torch.Tensor) -> torch.Tensor:
    other_logits = logits - onehot_like(logits, exclude, value=float("Inf"))
    return other_logits.argmax(dim=-1)


class TargetedMisclassificationWithGroundTruth(foolbox.criteria.TargetedMisclassification):
    def __init__(self, target_classes, ground_truth_classes):
        super().__init__(target_classes=target_classes)
        self.ground_truth_classes: ep.Tensor = ep.astensor(ground_truth_classes)
