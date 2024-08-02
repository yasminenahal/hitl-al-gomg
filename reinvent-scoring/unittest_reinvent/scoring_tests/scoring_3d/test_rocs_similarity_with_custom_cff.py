import unittest
import pytest

from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY_CFF, ROCS_CUSTOM_CFF
from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from unittest_reinvent.fixtures.test_data import AMOXAPINE, METHOXYHYDRAZINE, INVALID
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestRocsSimilarityWithCustomCff(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        csp_enum = ComponentSpecificParametersEnum()
        specific_parameters = {"shape_weight": 0.5, "color_weight": 0.5,
                               "similarity_measure": sim_measure_enum.REF_TVERSKY,
                               "rocs_input": ROCS_SHAPE_QUERY_CFF,
                               "input_type": input_type_enum.SHAPE_QUERY,
                               "custom_cff": ROCS_CUSTOM_CFF,
                               csp_enum.TRANSFORMATION: {},
                               "max_num_cpus": 8
                               }
        rocs_sim = component_parameters(component_type=sf_enum.ROCS_SIMILARITY, specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[rocs_sim])

    def test_parallel_rocs_similarity_1(self):
        score = self.sf_state.get_final_score(smiles=[AMOXAPINE])
        self.assertAlmostEqual(score.total_score, [0.42], delta=0.01)

    def test_parallel_rocs_similarity_2(self):
        score = self.sf_state.get_final_score(smiles=[METHOXYHYDRAZINE])
        self.assertAlmostEqual(score.total_score, [0.03], delta=0.01)

    def test_parallel_rocs_similarity_3(self):
        score = self.sf_state.get_final_score(smiles=[INVALID])
        self.assertEqual(score.total_score, [0.0])
