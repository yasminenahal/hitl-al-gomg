import unittest
import pytest
import numpy.testing as npt

from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY_3
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum, ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import COCAINE, CAFFEINE, CYCLODECANE, PARACETAMOL
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityTverskyScoreBug(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        specific_parameters = {rsp_enum.SHAPE_WEIGHT: 0.0, rsp_enum.COLOR_WEIGHT: 1.0,
                               rsp_enum.SIM_MEASURE: sim_measure_enum.REF_TVERSKY,
                               rsp_enum.ROCS_INPUT: ROCS_SHAPE_QUERY_3,
                               rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
                               rsp_enum.MAX_CPUS: 8
                               }
        rocs_sim = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                        name="parallel_rocs_similarity",
                                        specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[rocs_sim])

    def test_parallel_rocs_similarity(self):
        smiles = [COCAINE, CAFFEINE, CYCLODECANE, PARACETAMOL]
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_less(score.total_score, [1.0, 1.0, 1.0, 1.0])
