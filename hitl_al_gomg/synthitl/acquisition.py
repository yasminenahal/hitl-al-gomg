import numpy as np

from hitl_al_gomg.utils import ecfp_generator
from hitl_al_gomg.synthitl.simulated_expert import utility

fp_counter = ecfp_generator(radius=3, useCounts=True)


# Some functions were taken and adapted from Sundin et al. (2022) code (https://github.com/MolecularAI/reinvent-hitl)
def local_idx_to_fulldata_idx(N, selected_feedback, idx):
    all_idx = np.arange(N)
    mask = np.ones(N, dtype=bool)  # Boolean mask where selected indices will be False
    mask[selected_feedback] = False  # Mark the selected feedback as False in the mask
    pred_idx = all_idx[mask]  # The remaining unselected indices

    # Handle potential index issues by ensuring `idx` is valid
    try:
        return pred_idx[idx]  # Return the corresponding global index
    except IndexError:
        # If there are invalid indices (perhaps due to filtering), ensure valid index return
        valid_idx = [i if 0 <= i < len(pred_idx) else len(pred_idx) - 1 for i in idx]
        return pred_idx[valid_idx]

def epig(data, n, smiles, fit, selected_feedback, rng=None):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    fit: trained predictive model
    selected_feedback: previously selected in previous feedback rounds
    rng: random number generator
    """
    N = len(data)
    fps_pool = fp_counter.get_fingerprints(smiles)
    fps_target = fp_counter.get_fingerprints(smiles[:1000])
    probs_pool = fit._get_prob_distribution(fps_pool)
    probs_target = fit._get_prob_distribution(fps_target)
    estimated_epig_scores = fit._estimate_epig(probs_pool, probs_target)
    query_idx = np.argsort(estimated_epig_scores.numpy())[::-1][:n]  # Get the n highest
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)


def uncertainty_sampling(data, n, smiles, fit, selected_feedback, rng=None):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    fit: trained predictive model
    selected_feedback: previously selected in previous feedback rounds
    rng: random number generator
    """
    N = len(data)
    fps = fp_counter.get_fingerprints(smiles)
    estimated_unc = fit._uncertainty(fps)
    print(estimated_unc)
    query_idx = np.argsort(estimated_unc)[::-1][:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)


def entropy_based_sampling(data, n, smiles, fit, selected_feedback, rng=None):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    fit: trained predictive model
    selected_feedback: previously selected in previous feedback rounds
    rng: random number generator
    """
    N = len(data)
    fps = fp_counter.get_fingerprints(smiles)
    estimated_unc = fit._entropy(fps)
    query_idx = np.argsort(estimated_unc)[::-1][:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)


def exploitation_classification(
    data, n, smiles, fit, selected_feedback, rng=None
):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    fit: trained predictive model
    selected_feedback: previously selected in previous feedback rounds
    rng: random number generator
    """
    N = len(data)
    fps = fp_counter.get_fingerprints(smiles)
    score_pred = fit._predict_proba(fps)[:, 1]
    query_idx = np.argsort(score_pred)[::-1][:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)


def exploitation_regression(data, n, smiles, fit, selected_feedback, rng=None):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    fit: trained predictive model
    selected_feedback: previously selected in previous feedback rounds
    rng: random number generator
    """
    N = len(data)
    fps = fp_counter.get_fingerprints(smiles)
    values = fit._predict(fps)
    score_pred = [
        utility(v, low=2, high=4) for v in values
    ]  # Low and high values specified according to the paper experiment (use case 1)
    query_idx = np.argsort(score_pred)[::-1][:n]  # Get the n highest
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)


def margin_selection(data, n, smiles, fit, selected_feedback, rng=None):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    fit: trained predictive model
    selected_feedback: previously selected in previous feedback rounds
    rng: random number generator
    """
    N = len(data)
    fps = fp_counter.get_fingerprints(smiles)
    predicted_prob = fit._predict_proba(fps)
    rev = np.sort(predicted_prob, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    query_idx = np.argsort(values)[:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)


def random_selection(data, n, smiles, fit, selected_feedback, rng):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    fit: trained predictive model
    selected_feedback: previously selected in previous feedback rounds
    rng: random number generator
    """
    N = len(smiles)
    query_idx = rng.choice(N-len(selected_feedback),n, replace=False)
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)


def select_query(
    data, n, smiles, fit, selected_feedback, acquisition="random", rng=None
):
    """
    Parameters
    ----------
    smiles: array-like object of high-scoring smiles
    n: number of queries to select
    fit: trained predictive model
    acquisition: acquisition type
    rng: random number generator

    Returns
    -------
    int idx:
        Index of the query

    """
    if acquisition == "uncertainty":
        acq = uncertainty_sampling
    elif acquisition == "entropy":
        acq = entropy_based_sampling
    elif acquisition == "greedy_classification":
        acq = exploitation_classification
    elif acquisition == "greedy_regression":
        acq = exploitation_regression
    elif acquisition == "random":
        acq = random_selection
    elif acquisition == "epig":
        acq = epig
    elif acquisition == "margin":
        acq = margin_selection  # (Margin selection not tested in the paper experiments)
    else:
        print("Warning: unknown acquisition criterion. Using random sampling.")
        acq = random_selection
    return acq(data, n, smiles, fit, selected_feedback, rng)