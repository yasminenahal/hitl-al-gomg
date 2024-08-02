from typing import List

import numpy as np
from rdkit import Chem

from reinvent_scoring import ComponentParameters, TanimotoSimilarity, ScoringFunctionComponentNameEnum, JaccardDistance
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from unittest_reinvent.fixtures.test_data import BUTANE, CELECOXIB


def instantiate_component(smiles: List[str] = None, specific_parameters: dict = None):
    if specific_parameters is None:
        specific_parameters = {}
    if smiles is None:
        smiles = [BUTANE, CELECOXIB]
    specific_parameters['smiles'] = smiles
    return TanimotoSimilarity(ComponentParameters(
        component_type=ScoringFunctionComponentNameEnum().TANIMOTO_SIMILARITY,
        name="tanimoto_similarity",
        weight=1.,
        specific_parameters=specific_parameters))


def instantiate_jaccard_component(smiles: List[str] = None, specific_parameters: dict = None):
    if specific_parameters is None:
        specific_parameters = {}
    if smiles is None:
        smiles = [BUTANE, CELECOXIB]
    specific_parameters['smiles'] = smiles
    return JaccardDistance(
        ComponentParameters(
            component_type=ScoringFunctionComponentNameEnum().JACCARD_DISTANCE,
            name="jaccard_distance",
            weight=1.0,
            specific_parameters=specific_parameters,
        )
    )


def score(component: BaseScoreComponent, smiles: List[str]) -> np.array:
    """Calculates score for a list of SMILES strings."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    score = component.calculate_score(mols)
    return score.total_score


def score_single(component: BaseScoreComponent, smi: str) -> float:
    """Calculates score for a single SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    score = component.calculate_score([mol])
    return score.total_score[0]
