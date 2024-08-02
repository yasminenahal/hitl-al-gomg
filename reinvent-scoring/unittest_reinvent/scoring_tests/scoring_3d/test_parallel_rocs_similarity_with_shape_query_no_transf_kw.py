import unittest
import pytest
import numpy.testing as npt

from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from unittest_reinvent.fixtures.paths import  ROCS_SHAPE_QUERY
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum, ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum

from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE, INVALID
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityWithShapeQueryNoTransfKw(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        specific_parameters = {rsp_enum.SHAPE_WEIGHT: 0.5,
                               rsp_enum.COLOR_WEIGHT: 0.5,
                               rsp_enum.SIM_MEASURE: sim_measure_enum.REF_TVERSKY,
                               rsp_enum.ROCS_INPUT: ROCS_SHAPE_QUERY,
                               rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
                               rsp_enum.MAX_CPUS: 8
                               }
        rocs_sim = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                       name="parallel_rocs_similarity",
                                       specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[rocs_sim])

    def test_parallel_rocs_similarity_1(self):
        smiles = [CELECOXIB]*128
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score[0], [0.38], 2)

    def test_parallel_rocs_similarity_2(self):
        score = self.sf_state.get_final_score(smiles=[METAMIZOLE])
        self.assertAlmostEqual(score.total_score, [0.34], delta=0.01)

    def test_parallel_rocs_similarity_3(self):
        score = self.sf_state.get_final_score(smiles=[INVALID])
        self.assertEqual(score.total_score, [0.0])
