# Quantization routines proposed in Bonnet et al. "What if Adversarial Samples were Digital Images?", ACM IH&MMSec 2020. https://dl.acm.org/doi/10.1145/3369412.3395062
# This includes:
# (1) distortion-based quantization
# (2) gradient-based quantization
# (3) Lagrangian quantization
#
import numpy as np
import torch
from tqdm import tqdm


def distortion_based_quantization(x_org, x_adv):
    """
    Distortion-based quantization
    This quantization minimizes the distortion, i.e., the distortion must be lower or equal after the quantization.
    However, this quantization does not guarantee that the quantized image remains adversarial.
    :param x_org: original image with uint8 values in range [0., 255.], torch tensor of dtype float and shape [num_channels, height, width]
    :param x_adv: image found by adversarial attack with arbitrary float values in range [0., 255.], torch tensor of dtype float
    :return: quantized image with integer intensities in range [0., 255.], torch tensor of dtype float
    """
    # Eq. (2)
    unquantized_perturbation = x_adv - x_org

    # If the unquantized perturbation is an integer already, preserve this number.
    # Essentially, this step prevents floating point errors, e.g. prevent floor(0.9999) = 0.
    rounded_perturbation = torch.round(unquantized_perturbation)
    unquantized_perturbation = torch.where(
        torch.isclose(unquantized_perturbation, rounded_perturbation, atol=1e-3),
        rounded_perturbation,
        unquantized_perturbation)

    # Eq. (20)
    # If the attack reduces the pixel value, we round up towards the original value to minimize the distortion
    # If the attack increases the pixel value, we round down towards the original value to minimize the distortion
    x_quant = x_org + torch.where(
        unquantized_perturbation <= 0,
        torch.ceil(unquantized_perturbation),
        torch.floor(unquantized_perturbation))

    return x_quant


def gradient_based_quantization(x_org, y_true, x_adv, model, preprocessing=None):
    """
    Gradient-based optimization: Strengthen the adversarial properties of the image
    :param x_org: original image with uint8 values in range [0., 255.], torch tensor of dtype float and shape [num_channels, height, width]
    :param y_true: true class label of original image
    :param x_adv: image found by adversarial attack with arbitrary float values in range [0., 255.], torch tensor of dtype float
    :param model: PyTorch classifier
    :param preprocessing: transform for input preprocessing. Default is None.
    :return: quantized image with integer intensities in range [0., 255.], torch tensor of dtype float
    """

    # Eq. (2)
    unquantized_perturbation = x_adv - x_org

    # If the unquantized perturbation is an integer already, preserve this number.
    # Essentially, this step prevents floating point errors, e.g. prevent floor(0.9999) = 0.
    rounded_perturbation = torch.round(unquantized_perturbation)
    unquantized_perturbation = torch.where(
        torch.isclose(unquantized_perturbation, rounded_perturbation, atol=1e-3),
        rounded_perturbation,
        unquantized_perturbation)

    # Gradient of approximate loss w.r.t. inputs
    g = _obtain_gradient_approximation(y_true=y_true, x_adv=x_adv, model=model, preprocessing=preprocessing)

    # Eq. (25): Minimize loss through approximation, which is a correlation over the pixels.
    # The quantization noise is q[j] = {ceil(u[j]), floor(u[j])} - u(j)
    # To reduce the loss, we want q[j] * g[j] to be negative.
    # If g[j] < 0 => q[j] = ceil(u[j]) - u[j] > 0. => Ceil
    # If g[j] > 0 => g[j] = floor(u[j]) - u[j] < 0 => Floor
    # If g[j] = 0 => Minimize the distortion
    x_quant = torch.where(
        torch.isclose(g, torch.zeros(1, dtype=g.dtype, device=g.device)),
        distortion_based_quantization(x_org, x_adv),
        x_org + torch.where(g < 0,
                            torch.ceil(unquantized_perturbation),
                            torch.floor(unquantized_perturbation))
    )

    return x_quant


def _eval_attack_loss(x_quant, y_true, model, preprocessing=None, confidence=0.):
    """
    Compute attack loss
    :param x_quant: test image of shape [num_channels, height, width] as torch tensor
    :param y_true: correct label
    :param model: PyTorch module
    :param preprocessing: transform that operates on a single image
    :return: attack loss as torch tensor. This loss is negative if the attack succeeded
    """

    # Apply preprocessing if needed
    x_quant_preprocessed = x_quant
    if preprocessing is not None:
        x_quant_preprocessed = preprocessing(x_quant)

    # Temporarily prepend batch dimension
    x_quant_preprocessed_batch = torch.unsqueeze(x_quant_preprocessed, dim=0)

    # Feed through model
    y_pred_quant_batch = model(x_quant_preprocessed_batch)

    # Remove temporary batch dimension
    y_pred_quant = torch.squeeze(y_pred_quant_batch, dim=0)

    # Eq. (23): argmax_{k \neq t(x_o)} p_a(k)
    # Obtain the class label where the attack aimed to drive the sample x_adv, with or without success
    kappa_adv = _best_other_class(logits=y_pred_quant, y_true=y_true)

    # Eq. (22): p_q := f( a( x_adv + q) )
    p_q = y_pred_quant

    # Eq. (21): p_q(t(x_o)) - p_q(kappa_a)
    # The loss is the difference between the predicted probability that x_quant belongs to the true class of x_o minus the one of a given class kappa_a.
    # If the loss is smaller than zero, the attack succeeded.
    # This loss is independent of whatever loss the original attack used.
    L_q = p_q[y_true] - p_q[kappa_adv] + confidence

    return L_q


def _obtain_gradient_approximation(y_true, x_adv, model, q=None, preprocessing=None, confidence=0.):
    """
    Set up a loss function that encourages a misclassification. Obtain gradient w.r.t. input pixels by first-order approximation, which is a correlation over pixels.
    The first-order approximation is the Taylor expansion:
      f(x + a) = f(x) + df(x) / dx * a + 1/2 d^2 f(x) / d x^2 * a^2 + ...

    In our case:
      L(x_adv + q) = L(x_adv) + g^T q

    where g := grad L(x_adv) w.r.t. x_adv

    :param y_true: true class label of original image
    :param x_adv: image found by adversarial attack with arbitrary float values in range [0., 255.], torch tensor of dtype float
    :param model: PyTorch classifier
    :param q: Approximate loss at position x_adv + q. If q is not given, q will be set to zero.
    :param preprocessing: transform that applies preprocessing if needed
    :return: gradient w.r.t. image pixels
    """
    # Clone x_adv to ensure it is a leaf variable
    x_adv = x_adv.detach().clone()
    x_adv.requires_grad = True

    # Side note: Computing the gradient w.r.t. q is actually the same as w.r.t. x_adv
    if q is None:
        q = torch.zeros_like(x_adv, requires_grad=True)
    else:
        q = q.detach().clone()
        q.requires_grad = True

    # Eq. (3)
    x_quant = x_adv + q

    model.zero_grad()

    L_q = _eval_attack_loss(x_quant=x_quant, y_true=y_true, model=model, preprocessing=preprocessing, confidence=confidence)

    # Approximate loss using first-order Taylor expansion around (q = 0)
    # Eq. (24)
    L_q.backward()

    # Gradient of approximate loss w.r.t. inputs
    g = q.grad

    return g


def _quantize_lagrangian(x_org, unquantized_perturbation, g, lambda_):
    """
    Lagrangian quantization rule: Balance between distortion and attack
    :param x_org: original image with uint8 values in range [0., 255.], torch tensor of dtype float and shape [num_channels, height, width]
    :param unquantized_perturbation: difference between adversarial and original image
    :param g: attack loss gradient w.r.t. image pixels
    :param lambda_: hyper-parameter to trade off distortion and probability of successful attack
    :return: quantized image of shape [num_channels, height, width]
    """

    # Eq. (28)
    # By setting lambda = 0, we obtain the distortion-based quantization rule from Eq. (20)
    # By setting lambda -> +inf, this quantization routine converges to the gradient-based method from Eq. (25)
    lagrangian_perturbation = torch.where(
        1 - 2 * torch.ceil(unquantized_perturbation) > lambda_ * g,
        torch.ceil(unquantized_perturbation),
        torch.floor(unquantized_perturbation))

    distortion_based_perturbation = torch.where(
        unquantized_perturbation <= 0,
        torch.ceil(unquantized_perturbation),
        torch.floor(unquantized_perturbation)
    )

    # For pixels with gradient close to zero, use distortion-based quantization
    perturbation = torch.where(
        torch.isclose(g, torch.zeros(1, dtype=g.dtype, device=g.device)),
        distortion_based_perturbation,
        lagrangian_perturbation
    )

    # Add perturbation onto original image
    x_quant = x_org + perturbation

    return x_quant


def _best_other_class(logits, y_true):
    # Sort the indices in descending order
    most_likely_classes = torch.argsort(logits, descending=True)

    # Select the most likely class that is not the ground truth class
    if most_likely_classes[0] == y_true:
        kappa_adv = most_likely_classes[1]
    else:
        kappa_adv = most_likely_classes[0]

    return kappa_adv


def _eval_approximate_attack_loss(unquantized_attack_loss, g, q):
    return unquantized_attack_loss + torch.sum(g * q)


def _attack_succeeded(x_quant, y_true, model, preprocessing=None, confidence=0.):
    """
    Check whether a given example is misclassified by the desired confidence
    :param x_quant: test image of shape [num_channels, height, width]
    :param y_true: correct label
    :param model: PyTorch module
    :param preprocessing: transform that operates on a single image
    :return: True if loss from adversarial attack is smaller than 0
    """

    with torch.no_grad():
        is_adv_loss = _eval_attack_loss(
            x_quant=x_quant,
            y_true=y_true,
            model=model,
            preprocessing=preprocessing,
            confidence=confidence
        )

    return is_adv_loss < 0


def lagrangian_quantization_with_lambda(x_org, y_true, x_adv, model, lambda_):
    """
    Minimize a linear combination of the distortion and the classifier loss
    :param x_org: original image with uint8 values in range [0., 255.], torch tensor of dtype float and shape [num_channels, height, width]
    :param y_true: true class label of original image
    :param x_adv: image found by adversarial attack with arbitrary float values in range [0., 255.], torch tensor of dtype float
    :param model: PyTorch classifier
    :param lambda_: trade-off between distortion and classifier loss
    :return: quantized image with integer intensities in range [0., 255.], torch tensor of dtype float
    """
    # Eq. (2)
    unquantized_perturbation = x_adv - x_org

    # If the unquantized perturbation is an integer already, preserve this number.
    # Essentially, this step prevents floating point errors, e.g. prevent floor(0.9999) = 0.
    rounded_perturbation = torch.round(unquantized_perturbation)
    unquantized_perturbation = torch.where(
        torch.isclose(unquantized_perturbation, rounded_perturbation, atol=1e-3),
        rounded_perturbation,
        unquantized_perturbation)

    # Set up attack loss and approximate gradient of loss w.r.t. inputs
    # See Eq. (24)
    g = _obtain_gradient_approximation(y_true=y_true, x_adv=x_adv, model=model)

    x_quant = _quantize_lagrangian(x_org=x_org, unquantized_perturbation=unquantized_perturbation, g=g, lambda_=lambda_)

    return x_quant


def lagrangian_quantization(x_org, y_true, x_adv, model, binary_search_max_iter=100, preprocessing=None, confidence=0.):
    """
    Minimize a linear combination of the distortion and the classifier loss
    :param x_org: original image with uint8 values in range [0., 255.], torch tensor of dtype float and shape [num_channels, height, width]
    :param y_true: true class label of original image
    :param x_adv: image found by adversarial attack with arbitrary float values in range [0., 255.], torch tensor of dtype float
    :param model: PyTorch classifier
    :param binary_search_max_iter: maximum number of iterations to run for binary search over lambda
    :param preprocessing: transform for input preprocessing. Default is None.
    :param confidence: Desired confidence level (similar to confidence in C&W attack). Default is 0.
    :return: quantized image with integer intensities in range [0., 255.], torch tensor of dtype float
    """

    # If the unquantized perturbation is an integer already, preserve this number.
    # Essentially, prevent floating point errors, e.g., prevent floor(0.9999) = 0.
    x_adv_rounded = torch.round(x_adv)
    x_adv = torch.where(torch.isclose(x_adv, x_adv_rounded, atol=1e-4), x_adv_rounded, x_adv)

    # Eq. (2)
    unquantized_perturbation = x_adv - x_org

    # Set up attack loss and approximate gradient of loss w.r.t. inputs
    # See Eq. (24)
    g = _obtain_gradient_approximation(y_true=y_true, x_adv=x_adv, model=model, preprocessing=preprocessing)

    # Define set of indices whose quantization depends on lambda
    # Eq. (29)
    quantization_depends_on_lambda_mask = torch.logical_and(
        ~torch.isclose(g, torch.zeros(1, dtype=g.dtype, device=g.device)),
        torch.sign(g) != torch.sign(torch.ceil(unquantized_perturbation) - 0.5))

    # If no pixel depends on lambda, return distortion quantization
    if torch.sum(quantization_depends_on_lambda_mask) == 0:
        x_distortion_quant = x_org + torch.where(unquantized_perturbation <= 0,
                                                 torch.ceil(unquantized_perturbation),
                                                 torch.floor(unquantized_perturbation))
        return x_distortion_quant

    # Rewrite quantization rule
    # If r[j] > lambda, then ceil, and otherwise floor
    r = (1. - 2 * torch.ceil(unquantized_perturbation[quantization_depends_on_lambda_mask])) / g[quantization_depends_on_lambda_mask]

    # Sort ratios in ascending order
    # Pixels ranked first offer a better trade-off: they yield a valuable loss decrease for a modest distortion increase.
    r = torch.sort(r)[0]

    # As candidates for lambda, we need to explore the interval [0, max_j r[j]].
    lower_limit = 0
    upper_limit = torch.max(r).item()

    # Early stopping
    # Check whether setting the maximum value for lambda leads to an adversarial image
    x_quant = _quantize_lagrangian(x_org=x_org, unquantized_perturbation=unquantized_perturbation, g=g, lambda_=upper_limit)
    if not _attack_succeeded(x_quant=x_quant, y_true=y_true, model=model, preprocessing=preprocessing, confidence=confidence):
        raise ValueError("Even after setting lambda to the maximum value, the image is not an adversarial example")

    # Check whether setting the minimum value for lambda is already an adversarial image
    x_quant = _quantize_lagrangian(x_org=x_org, unquantized_perturbation=unquantized_perturbation, g=g, lambda_=lower_limit)
    if _attack_succeeded(x_quant=x_quant, y_true=y_true, model=model, preprocessing=preprocessing, confidence=confidence):
        # We're done. This yields the minimum distortion possible
        return x_quant

    # Use binary search to choose the smallest lambda that gives rise to an adversarial image
    converged = False
    next_iter = 0
    while not converged:
        lambda_candidate = (lower_limit + upper_limit) / 2

        x_quant = _quantize_lagrangian(x_org=x_org, unquantized_perturbation=unquantized_perturbation, g=g, lambda_=lambda_candidate)
        if _attack_succeeded(x_quant=x_quant, y_true=y_true, model=model, preprocessing=preprocessing, confidence=confidence):
            upper_limit = lambda_candidate
        else:
            lower_limit = lambda_candidate

        next_iter += 1

        if next_iter >= binary_search_max_iter or np.isclose(lower_limit, upper_limit, atol=1e-3):
            break

    x_quant = _quantize_lagrangian(x_org=x_org, unquantized_perturbation=unquantized_perturbation, g=g, lambda_=upper_limit)
    assert _attack_succeeded(x_quant, y_true=y_true, model=model, preprocessing=preprocessing, confidence=confidence)
    return x_quant


def lagrangian_quantization_ddn(x_org, y_true, x_adv, model, perturbation_budget, was_prev_x_adv_successful, preprocessing=None, confidence=0., verbose=0, iteration=-1):
    """
    Minimize a linear combination of the distortion and the classifier loss
    :param x_org: original image with uint8 values in range [0., 255.], torch tensor of dtype float and shape [num_channels, height, width]
    :param y_true: true class label of original image
    :param x_adv: image found by adversarial attack with arbitrary float values in range [0., 255.], torch tensor of dtype float
    :param model: PyTorch classifier
    :param perturbation_budget: maximum perturbation after quantization (squared sum of pixel changes)
    :param was_prev_x_adv_successful: boolean flag that indicates whether x_adv from the previous iteration was adversarial
    :param preprocessing: transform for input preprocessing. Default is None.
    :return: quantized image with integer intensities in range [0., 255.], torch tensor of dtype float
    """

    # Prevent rounding errors
    x_org_rounded = torch.round(x_org)
    assert torch.allclose(x_org, x_org_rounded)
    x_org = x_org_rounded

    # If the unquantized perturbation is an integer already up to some precision, preserve this value.
    # Essentially, prevent floating point errors, e.g., prevent floor(0.9999) = 0.
    x_adv_rounded = torch.round(x_adv)
    x_adv = torch.where(torch.isclose(x_adv, x_adv_rounded, atol=1e-4), x_adv_rounded, x_adv)

    # Eq. (2)
    unquantized_perturbation = x_adv - x_org

    # What is the perturbation cost if we always round back towards the original image?
    # This is a minimum perturbation cost that we can achieve.
    # Eq. (20)
    x_distortion_quant = x_org + torch.where(unquantized_perturbation <= 0, torch.ceil(unquantized_perturbation), torch.floor(unquantized_perturbation))

    distortion_quant_per_pixel_cost = torch.square(x_distortion_quant - x_org)
    # This is the first share of our perturbation budget.
    perturbation_cost = torch.sum(distortion_quant_per_pixel_cost)

    # If distortion-based quantization cannot stay within the distortion budget, we won't find anything better
    # The L2 perturbation is computed as sqrt(sum(square(delta)). Remove the sqrt
    perturbation_budget_squared = perturbation_budget ** 2
    if perturbation_cost >= perturbation_budget_squared:
        return x_distortion_quant

    # Set up attack loss and approximate gradient of loss w.r.t. inputs
    # See Eq. (24)
    g = _obtain_gradient_approximation(y_true=y_true, x_adv=x_adv, model=model, preprocessing=preprocessing, confidence=confidence)

    # For some pixels, we round towards the original sample anyway.
    # This is the case when one quantization value minimizes both the distortion and the classifier loss.
    # Define set of indices whose quantization depends on lambda
    # Eq. (29)
    grad_close_to_zero = torch.isclose(g, torch.zeros(1, dtype=g.dtype, device=g.device))
    quantization_depends_on_lambda_mask = torch.logical_and(
        ~grad_close_to_zero,
        torch.sign(g) != torch.sign(torch.ceil(unquantized_perturbation) - 0.5))

    # Additionally, when the unquantized perturbation is a real number already, it does not depend on lambda
    # The paper assumes that this special case rarely occurs.
    quantization_depends_on_lambda_mask = torch.logical_and(
        quantization_depends_on_lambda_mask,
        ~torch.isclose(unquantized_perturbation, torch.round(unquantized_perturbation), atol=1e-4),
    )

    # If no pixel depends on lambda, return distortion-based quantization
    if torch.sum(quantization_depends_on_lambda_mask) == 0:
        return x_distortion_quant

    # What would be the cost for gradient-based quantization?
    # Eq. (25): Minimize loss through approximation, which is a correlation over the pixels.
    # The quantization noise is q[j] = {ceil(u[j]), floor(u[j])} - u(j)
    # To reduce the loss, we want q[j] * g[j] to be negative.
    # If g[j] < 0 => q[j] = ceil(u[j]) - u[j] > 0. => Ceil
    # If g[j] > 0 => g[j] = floor(u[j]) - u[j] < 0 => Floor
    # If g[j] = 0 => Minimize the distortion
    x_gradient_quant = torch.where(
        grad_close_to_zero,
        x_distortion_quant,
        x_org + torch.where(g < 0,
                            torch.ceil(unquantized_perturbation),
                            torch.floor(unquantized_perturbation))
    )

    # How much perturbation budget does gradient-based quantization cost?
    gradient_quant_per_pixel_cost = torch.square(x_gradient_quant - x_org)

    # For each pixel, if we switch from distortion-based to gradient-based quantization, this is the additional cost that needs to be payed
    additional_per_pixel_cost = gradient_quant_per_pixel_cost - distortion_quant_per_pixel_cost
    assert torch.all(additional_per_pixel_cost[quantization_depends_on_lambda_mask] > 0)

    # Rewrite quantization rule
    # If r[j] > lambda, then ceil, and otherwise floor
    r = (1 - 2 * torch.ceil(unquantized_perturbation[quantization_depends_on_lambda_mask])) / g[quantization_depends_on_lambda_mask]
    assert torch.all(r > 0)

    # Sort ratios in ascending order
    # Pixels ranked first offer a better trade-off: they yield a valuable loss decrease for a modest distortion increase.
    r_sorted, r_indices = torch.sort(r)

    # What is the maximum value of r such that we stay within our distortion budget?
    # Re-order per-pixel cost based on ranking
    additional_per_pixel_cost_ranked = additional_per_pixel_cost[quantization_depends_on_lambda_mask][r_indices]
    # Cumulative sum of ranked distortion
    additional_per_pixel_cost_cumsum = torch.cumsum(additional_per_pixel_cost_ranked, dim=0)

    # If x_adv in the previous iteration was adversarial:
    if was_prev_x_adv_successful:
        # Find the last index k^* s.t. delta^* = r(k^*) produces distortion smaller than budget
        possible_ks = torch.where(perturbation_cost + additional_per_pixel_cost_cumsum <= perturbation_budget_squared)[0]
        if 0 == len(possible_ks):
            # Example for a special case: perturbation_cost = 0, perturbation_budget_squared = 0.9
            return x_distortion_quant

        k = possible_ks[-1] + 1

    else:
        # x_adv is not yet adversarial
        # If the perturbation budget is larger than gradient-based quantization, we can select the maximum k
        if perturbation_cost + additional_per_pixel_cost_cumsum[-1] <= perturbation_budget_squared:
            k = len(r_sorted)

        else:
            # Find the first index k^* s.t. delta^* = r(k^*) produces distortion bigger than budget
            possible_ks = torch.where(perturbation_cost + additional_per_pixel_cost_cumsum > perturbation_budget_squared)[0]
            k = possible_ks[0] + 1

    # Variant 1: Quantize based on lambda
    delta = r_sorted[k - 1].item()
    # x_quant = _quantize_lagrangian(x_org=x_org, unquantized_perturbation=unquantized_perturbation, g=g, lambda_=delta)

    if verbose > 1:
        plot_attack_loss_vs_distortion(
            x_org=x_org,
            unquantized_perturbation=unquantized_perturbation,
            g=g,
            lambdas=r_sorted,
            y_true=y_true,
            model=model,
            preprocessing=preprocessing,
            confidence=confidence,
            selected_lambda=delta,
            iteration=iteration,
            perturbation_budget_squared=perturbation_budget_squared.item(),
        )

    # Variant 2:
    # Use distortion-based quantization as baseline
    x_quant = x_distortion_quant.detach().clone()

    # Swap in gradient-based quantization at as many positions as the perturbation budget allows
    z_indices, y_indices, x_indices = torch.where(quantization_depends_on_lambda_mask)
    z_indices = z_indices[r_indices[:k]]
    y_indices = y_indices[r_indices[:k]]
    x_indices = x_indices[r_indices[:k]]
    x_quant[z_indices, y_indices, x_indices] = x_gradient_quant[z_indices, y_indices, x_indices]

    return x_quant


def plot_attack_loss_vs_distortion(x_org, unquantized_perturbation, g, lambdas, y_true, model, selected_lambda, perturbation_budget_squared, preprocessing=None, confidence=0., iteration=-1, output_dir=None):
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import os

    # Evaluate true attack loss at unquantized x_adv
    x_adv = x_org + unquantized_perturbation
    with torch.no_grad():
        unquantized_attack_loss = _eval_attack_loss(x_quant=x_adv, y_true=y_true, model=model, preprocessing=preprocessing, confidence=confidence)

    # Foolbox sums over square pixel differences (instead of averaging)
    x_adv_l2_distortion = torch.flatten(x_adv - x_org, start_dim=1).square().sum().sqrt().item()

    # Iterate over different values for lambda
    # Select up to 100 values for lambda
    num_lambdas = len(lambdas)
    lambda_indices = np.linspace(0, num_lambdas - 1, num=min(100, num_lambdas))
    lambda_candidates = lambdas[lambda_indices].detach().cpu().numpy()
    # Insert lambda selected based on the perturbation budget into the candidates that are evaluated
    ii = np.searchsorted(lambda_candidates, selected_lambda)
    lambda_candidates = np.insert(lambda_candidates, ii, selected_lambda)

    # For each lambda, evaluate true attack loss, approximate attack loss, and L2 distortion
    true_attack_losses = []
    approximate_attack_losses = []
    l2_distortions = []

    for l in tqdm(lambda_candidates, desc="lambda"):
        # Quantize using current lambda candidate
        x_quant = _quantize_lagrangian(x_org=x_org, unquantized_perturbation=unquantized_perturbation, g=g, lambda_=l)

        # Compute true attack loss
        with torch.no_grad():
            true_attack_loss = _eval_attack_loss(x_quant=x_quant, y_true=y_true, model=model, preprocessing=preprocessing, confidence=confidence).item()

        # Compute approximate attack loss
        q = x_quant - x_adv
        approximate_attack_loss = _eval_approximate_attack_loss(unquantized_attack_loss=unquantized_attack_loss, g=g, q=q).item()

        # Evaluate L2 distortion
        l2_distortion = torch.flatten(x_quant - x_org, start_dim=1).square().sum().sqrt().item()

        true_attack_losses.append(true_attack_loss)
        l2_distortions.append(l2_distortion)
        approximate_attack_losses.append(approximate_attack_loss)

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))

    ax.plot(lambda_candidates, true_attack_losses, color="b", label="True attack loss")
    ax.plot(lambda_candidates, approximate_attack_losses, color="g", label="Approximate attack loss")
    ax.plot(selected_lambda, true_attack_losses[ii], color="b", marker="x")
    ax.plot(selected_lambda, approximate_attack_losses[ii], color="g", marker="x")
    ax.axhline(y=0, linestyle="dashed", color="k")
    ax.set_title(f"Iteration {iteration:03d}, Unquantized perturbation: {x_adv_l2_distortion:4.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Loss")

    perturbation_budget = np.sqrt(perturbation_budget_squared)
    distortion_ax = ax.twinx()
    distortion_ax.plot(lambda_candidates, l2_distortions, color="r", label="L2 distortion")
    distortion_ax.plot(selected_lambda, l2_distortions[ii], color="r", marker="x")
    distortion_ax.axhline(y=perturbation_budget, linestyle="dashed", color="r", label="Perturbation budget")
    distortion_ax.set_ylabel("L2 distortion")

    fig.tight_layout()

    if output_dir is not None:
        filename = "lagrangian_quantization_iteration_{:03d}.png".format(iteration)
        fig.savefig(os.path.join(output_dir, filename), dpi=300)

    fig.show()
