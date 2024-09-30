import torch

import logging
import math


# Functions were originally implemented by Bickford Smith et al. (2023) to compute EPIG scores (https://github.com/fbickfordsmith/epig)
def conditional_epig_from_probs(
    probs_pool: torch.Tensor, probs_targ: torch.Tensor, batch_size: int = 100
) -> torch.Tensor:
    """
    See conditional_epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        batch_size: int, size of the batch to process at a time

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Get sizes
    N_p, K, Cl = probs_pool.size()
    N_t = probs_targ.size(0)

    # Prepare tensors
    scores = torch.zeros(N_p, N_t)

    # Process in batches to save memory
    for i in range(0, N_p, batch_size):
        for j in range(0, N_t, batch_size):
            # Get the batch
            probs_pool_batch = probs_pool[i : i + batch_size]
            probs_targ_batch = probs_targ[j : j + batch_size]

            # Estimate the joint predictive distribution.
            probs_pool_batch = probs_pool_batch.permute(1, 0, 2)  # [K, batch_size, Cl]
            probs_targ_batch = probs_targ_batch.permute(1, 0, 2)  # [K, batch_size, Cl]
            probs_pool_batch = probs_pool_batch[
                :, :, None, :, None
            ]  # [K, batch_size, 1, Cl, 1]
            probs_targ_batch = probs_targ_batch[
                :, None, :, None, :
            ]  # [K, 1, batch_size, 1, Cl]
            probs_pool_targ_joint = probs_pool_batch * probs_targ_batch
            probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)

            # Estimate the marginal predictive distributions.
            probs_pool_batch = torch.mean(probs_pool_batch, dim=0)
            probs_targ_batch = torch.mean(probs_targ_batch, dim=0)

            # Estimate the product of the marginal predictive distributions.
            probs_pool_targ_indep = probs_pool_batch * probs_targ_batch

            # Estimate the conditional expected predictive information gain for each pair of examples.
            # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
            nonzero_joint = probs_pool_targ_joint > 0
            log_term = torch.clone(probs_pool_targ_joint)
            log_term[nonzero_joint] = torch.log(probs_pool_targ_joint[nonzero_joint])
            log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])
            score_batch = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))

            # Store the results
            scores[i : i + batch_size, j : j + batch_size] = score_batch

    return scores  # [N_p, N_t]


def conditional_epig_from_values(
    values_pool: torch.Tensor,
    values_targ: torch.Tensor,
    batch_size: int = 1000,
) -> torch.Tensor:
    """
    Calculate conditional EPIG (Expected Predictive Information Gain)
    from continuous regression values.

    Arguments:
        values_pool: Tensor[float], [N_p, K]
            Continuous regression values for the pool set.
        values_targ: Tensor[float], [N_t, K]
            Continuous regression values for the target set.

    Returns:
        Tensor[float], [N_p, N_t]
            Conditional EPIG scores.
    """
    targ_mean = torch.mean(values_targ, dim=1)
    targ_mean = targ_mean.reshape(1, -1)

    num_samples_pool = values_pool.shape[0]

    scores_list = []

    for i in range(0, num_samples_pool, batch_size):

        values_pool_batch = values_pool[i : i + batch_size]

        # Estimate the joint predictive distribution.
        joint_mean_batch = torch.matmul(values_pool_batch, values_targ.unsqueeze(2))

        # Estimate the marginal predictive distributions.
        pool_mean_batch = torch.mean(values_pool_batch, dim=1)

        pool_mean_batch = pool_mean_batch.reshape(-1, 1)

        # Estimate the product of the marginal predictive distributions.
        indep_mean = pool_mean_batch * targ_mean

        # Estimate the conditional expected predictive information gain for each pair of examples.
        # This is the KL divergence between the joint predictive distribution and the product of the marginal predictive distributions.
        scores_list.append(
            torch.sum(
                joint_mean_batch
                * (torch.log(joint_mean_batch) - torch.log(indep_mean)),
                dim=1,
            )
        )

    scores = torch.cat(scores_list, dim=0)

    return scores


def conditional_epig_from_continuous(
    pred_pool: torch.Tensor, pred_targ: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the mean squared error (MSE) between the predicted values for pairs of examples.
    Suitable for regression models.

    Arguments:
        predictions_pool: Tensor[float], [N_p]
        predictions_targ: Tensor[float], [N_t]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Reshape pred_pool and pred_targ to have 2D shape for broadcasting
    pred_pool = pred_pool.unsqueeze(1)  # [N_p, 1]
    pred_targ = pred_targ.unsqueeze(0)  # [1, N_t]

    # Calculate the joint predictive distribution for all pairs of examples
    joint_pred_dist = pred_pool - pred_targ  # [N_p, N_t]

    # Calculate the conditional expected predictive information gain
    scores = joint_pred_dist**2

    print(len(pred_pool), len(pred_targ))
    print(joint_pred_dist.shape)
    print(scores.shape)
    return scores  # [N_p, N_t]


def check(
    scores: torch.Tensor,
    max_value: float = math.inf,
    epsilon: float = 1e-6,
    score_type: str = "",
) -> torch.Tensor:
    """
    Warn if any element of scores is negative, a nan or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores + epsilon >= 0) & (scores - epsilon <= max_value)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()

        logging.warning(
            f"Invalid {score_type} score (min = {min_score}, max = {max_score})"
        )

    return scores


def epig_from_conditional_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Arguments:
        scores: Tensor[float], [N_p, N_t]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, score_type="EPIG")  # [N_p,]
    return scores  # [N_p,]


def epig_from_probs(
    probs_pool: torch.Tensor, probs_targ: torch.Tensor, classification: str = True
) -> torch.Tensor:
    """
    See epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    if classification:
        scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    else:
        scores = conditional_epig_from_values(probs_pool, probs_targ)  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]
