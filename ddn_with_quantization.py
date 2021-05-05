#
# This method extends the Foolbox implementation by a confidence parameter and the quantization step: https://github.com/bethgelab/foolbox/blob/92af8673ee957e5af1f9ad5f3abb474c63f93397/foolbox/attacks/ddn.py
# Original implementation: https://github.com/jeromerony/adversarial-library/blob/main/adv_lib/attacks/decoupled_direction_norm.py
#
from foolbox.models import Model

from foolbox.criteria import Misclassification, TargetedMisclassification

from foolbox.distances import l2

from foolbox.devutils import atleast_kd, flatten

from foolbox.attacks.base import MinimizationAttack
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import T
from foolbox.attacks.base import raise_if_kwargs
from tqdm import tqdm
from lagrangian_quantization import lagrangian_quantization_ddn
from criteria import TargetedMisclassificationWithGroundTruth


from typing import Union, Tuple, Optional, Any
import math
import eagerpy as ep
import torch


NAIVE_ROUND = "naive_round"
LAGRANGIAN_QUANTIZATION = "lagrangian_quantization"
NO_QUANTIZATION = "no_quantization"


def normalize_gradient_l2_norms(grad: ep.Tensor) -> ep.Tensor:
    # Calculate norm of gradient
    norms = ep.norms.l2(flatten(grad), -1)

    # Remove zero gradients
    # Replace zero gradients by drawing from a normal distribution
    grad = ep.where(
        atleast_kd(norms == 0, grad.ndim), ep.normal(grad, shape=grad.shape), grad
    )
    # calculate norms again for previously vanishing elements
    norms = ep.norms.l2(flatten(grad), -1)

    norms = ep.maximum(norms, 1e-12)  # avoid division by zero
    factor = 1. / norms
    factor = atleast_kd(factor, grad.ndim)
    return grad * factor


class DDNQuantizationAttack(MinimizationAttack):
    """The Decoupled Direction and Norm L2 adversarial attack. [#Rony18]_

    Args:
        init_epsilon : Initial value for the norm/epsilon ball.
        steps : Number of steps for the optimization.
        gamma : Factor by which the norm will be modified: new_norm = norm * (1 + or - gamma).

    References:
        .. [#Rony18] Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed,
            Robert Sabourin, Eric Granger, "Decoupling Direction and Norm for
            Efficient Gradient-Based L2 Adversarial Attacks and Defenses",
            https://arxiv.org/abs/1811.09600
    """

    distance = l2

    def __init__(
        self, *, model_torch, preprocessing=None, init_epsilon: float = 1.0, steps: int = 10, gamma: float = 0.05, confidence: float = 0, verbose: int = 0, quantization: str = LAGRANGIAN_QUANTIZATION
    ):
        """
        Expects input images in range [0, 1]
        :param model_torch: PyTorch module
        :param preprocessing: (Optional) transform that preprocesses images in range [0, 255] to the input range of the model. Used by Lagrangian quantization.
        :param init_epsilon: Initial value for the norm of the attack's epsilon ball
        :param steps: number of optimization steps
        :param gamma: factor to modify the norm in each iteration
        :param confidence: The attack succeeds after enforcing the given margin between the logits corresponding to the wrong class and the true class. Set to 0 by default.
        """
        assert quantization in {LAGRANGIAN_QUANTIZATION, NAIVE_ROUND, NO_QUANTIZATION}, "Unknown quantization method"

        self.model_torch = model_torch
        self.preprocessing = preprocessing
        self.init_epsilon = init_epsilon
        self.steps = steps
        self.gamma = gamma
        self.confidence = confidence
        self.verbose = verbose
        self.quantization = quantization

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
            change_classes_logits = -self.confidence
        else:
            raise ValueError("unsupported criterion")

        def _is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            # Implementation analogously to Foolbox's Carlini & Wagner attack

            # For binary classification, this method checks whether
            #   logits[label_true] + confidence < y_pred[label_other]
            # or equivalently
            #   logits[label_true] - logits[label_other] + confidence < 0
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        max_stepsize = 1.0
        min_, max_ = model.bounds

        def loss_fn(
            inputs: ep.Tensor, labels: ep.Tensor
        ) -> Tuple[ep.Tensor, ep.Tensor]:
            logits = model(inputs)

            # In the untargeted case, our goal is to maximize this loss
            sign = -1.0 if targeted else 1.0
            loss = sign * ep.crossentropy(logits, labels).sum()

            return loss, logits

        grad_and_logits = ep.value_and_grad_fn(x, loss_fn, has_aux=True)

        delta = ep.zeros_like(x)

        epsilon = self.init_epsilon * ep.ones(x, len(x))
        worst_norm = ep.norms.l2(flatten(ep.maximum(x - min_, max_ - x)), -1)

        best_l2 = worst_norm
        best_delta = delta
        adv_found = ep.zeros(x, len(x)).bool()

        for i in tqdm(range(self.steps), desc="Optimizing epsilon", disable=self.verbose < 1):
            # perform cosine annealing of LR starting from 1.0 to 0.01
            # Denotes as alpha in the paper
            stepsize = (
                0.01
                + (max_stepsize - 0.01) * (1 + math.cos(math.pi * i / self.steps)) / 2
            )

            x_adv = x + delta

            _, logits, gradients = grad_and_logits(x_adv, classes)

            # Normalize gradient to unit length
            gradients_normalized = normalize_gradient_l2_norms(gradients)
            is_adversarial = _is_adversarial(x_adv, logits)

            # Calculate L2 distance in attack space
            l2 = ep.norms.l2(flatten(delta), axis=-1)
            is_smaller = l2 <= best_l2

            is_both = ep.logical_and(is_adversarial, is_smaller)
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_l2 = ep.where(is_both, l2, best_l2)

            best_delta = ep.where(atleast_kd(is_both, x.ndim), delta, best_delta)

            # Maximize loss function by walking into direction of steepest ascent
            delta = delta + stepsize * gradients_normalized

            # Adjust epsilon sphere
            # If we are not yet adversarial, increase the radius
            # If we obtained an adversarial examples, decrease the radius
            epsilon = epsilon * ep.where(
                is_adversarial, 1.0 - self.gamma, 1.0 + self.gamma
            )
            epsilon = ep.minimum(epsilon, worst_norm)

            # Project delta to epsilon ball
            delta *= atleast_kd(epsilon / ep.norms.l2(flatten(delta), -1), x.ndim)

            # Clip to valid bounds
            delta = ep.clip(x + delta, *model.bounds) - x

            # Epsilon is the perturbation budget
            perturbation_budget = epsilon * (max_ - min_)
            # The current distortion may be smaller after clipping to the model bounds
            assert torch.all((ep.norms.l2(flatten(delta), -1) <= perturbation_budget + 1e-3).raw)

            if NAIVE_ROUND == self.quantization:
                # Naive rounding
                # Ensure that multiplying the perturbation by 255 yields integer values
                # Note that after rounding, the quantization budget epsilon is almost always exceeded
                delta.raw.mul_(255).round_().div_(255)

            elif LAGRANGIAN_QUANTIZATION == self.quantization:
                for img_idx in range(N):
                    # Lagrangian quantization expects images in range [0, 255]
                    x_org = x[img_idx].raw * 255
                    x_adv_unquantized = (x + delta)[img_idx].raw * 255

                    # Get y_true and y_target for untargeted and targeted attacks
                    y_target = None
                    if isinstance(criterion_, Misclassification):
                        y_true = criterion_.labels.raw[img_idx]

                    elif isinstance(criterion_, TargetedMisclassificationWithGroundTruth):
                        y_true = criterion_.ground_truth_classes.raw[img_idx]
                        y_target = criterion_.target_classes.raw[img_idx]

                    else:
                        raise ValueError("Not implemented")

                    # Transform perturbation budget from images in [0, 1] to images in [0, 255]
                    transformed_perturbation_budget = perturbation_budget[img_idx].raw * 255
                    assert (x_org - x_adv_unquantized).square().sum().sqrt() <= transformed_perturbation_budget + 1e-3

                    x_quant = lagrangian_quantization_ddn(
                        x_org=x_org,
                        y_true=y_true,
                        y_target=y_target,
                        x_adv=x_adv_unquantized,
                        model=self.model_torch,
                        perturbation_budget=transformed_perturbation_budget,
                        was_prev_x_adv_successful=is_adversarial[img_idx].raw,
                        preprocessing=self.preprocessing,
                        confidence=self.confidence,
                        iteration=i,
                        verbose=self.verbose,
                    )

                    delta_after_quantization = (x_quant - x_org) / 255.
                    delta.raw[img_idx] = delta_after_quantization

            elif NO_QUANTIZATION == self.quantization:
                pass

            else:
                raise ValueError("Unknown quantization")

        x_adv = x + best_delta

        return restore_type(x_adv)
