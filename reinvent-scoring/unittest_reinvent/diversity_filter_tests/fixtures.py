import numpy as np
from typing import List

from reinvent_scoring.scoring.score_summary import FinalSummary, ComponentSummary
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


def tanimoto_scaffold_filter_arrangement(smiles: List[str], scores: List[float],
                                         valid_idx: List[int]) -> FinalSummary:
    component_parameters = ComponentParameters(
        component_type=ScoringFunctionComponentNameEnum().TANIMOTO_SIMILARITY,
        name="tanimoto_similarity",
        weight=1.,
        specific_parameters={})

    component_score_summary = ComponentSummary(scores, component_parameters)
    return FinalSummary(np.array(scores), smiles, valid_idx, [component_score_summary])