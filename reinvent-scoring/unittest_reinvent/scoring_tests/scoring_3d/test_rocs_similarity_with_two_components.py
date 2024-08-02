import unittest
import pytest

from typing import Dict, Any
from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY, ROCS_SHAPE_QUERY_2
from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestRocsSimilarityWithTwoComponents(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        specific_parameters_1 = self.specific_parameters(0.2, 0.8, ROCS_SHAPE_QUERY)
        rocs_sim_1 = component_parameters(component_type=sf_enum.ROCS_SIMILARITY,
                                          specific_parameters=specific_parameters_1)
        specific_parameters_2 = self.specific_parameters(0.5, 0.5, ROCS_SHAPE_QUERY_2)
        rocs_sim_2 = component_parameters(component_type=sf_enum.ROCS_SIMILARITY,
                                          specific_parameters=specific_parameters_2,
                                          name="rocs_similarity second query", )
        self.sf_state = CustomSum(parameters=[rocs_sim_1, rocs_sim_2])

    @staticmethod
    def specific_parameters(shape_weight: int, color_weight: int, rocs_input: str) -> Dict[Any, Any]:
        return {
            "shape_weight": shape_weight, "color_weight": color_weight,
            "similarity_measure":  ROCSSimilarityMeasuresEnum().REF_TVERSKY,
            "rocs_input": rocs_input,
            "input_type": ROCSInputFileTypesEnum().SHAPE_QUERY,
            ComponentSpecificParametersEnum().TRANSFORMATION: False
         }

    def test_rocs_similarity_1(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score, [0.41], delta=0.01)

    def test_rocs_similarity_2(self):
        score = self.sf_state.get_final_score(smiles=[METAMIZOLE])
        self.assertAlmostEqual(score.total_score[0], [0.43], delta=0.01)


