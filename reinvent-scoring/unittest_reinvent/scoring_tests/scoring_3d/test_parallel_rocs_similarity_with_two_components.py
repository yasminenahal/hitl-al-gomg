import unittest
import pytest

from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY, ROCS_SHAPE_QUERY_2
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum, ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityWithTwoComponents(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        specific_parameters_1 = {rsp_enum.SHAPE_WEIGHT: 0.2,
                                 rsp_enum.COLOR_WEIGHT: 0.8,
                                 rsp_enum.SIM_MEASURE: sim_measure_enum.REF_TVERSKY,
                                 rsp_enum.ROCS_INPUT: ROCS_SHAPE_QUERY,
                                 rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
                                 rsp_enum.MAX_CPUS: 8,
                                 }
        rocs_sim_1 = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                          name="parallel_rocs_similarity",
                                          specific_parameters=specific_parameters_1)
        specific_parameters_2 = {rsp_enum.SHAPE_WEIGHT: 0.5,
                                 rsp_enum.COLOR_WEIGHT: 0.5,
                                 rsp_enum.SIM_MEASURE: sim_measure_enum.REF_TVERSKY,
                                 rsp_enum.ROCS_INPUT: ROCS_SHAPE_QUERY_2,
                                 rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
                                 rsp_enum.MAX_CPUS: 8
                                 }
        rocs_sim_2 = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                          name="parallel_rocs_similarity",
                                          specific_parameters=specific_parameters_2)
        self.sf_state = CustomSum(parameters=[rocs_sim_1, rocs_sim_2])

    def test_parallel_rocs_similarity_1(self):
        smiles = [CELECOXIB]
        score = self.sf_state.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score, [0.41], delta=0.01)

    def test_parallel_rocs_similarity_2(self):
        score = self.sf_state.get_final_score(smiles=[METAMIZOLE])
        self.assertAlmostEqual(score.total_score, [0.40], delta=0.01)

