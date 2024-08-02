import unittest
import pytest

from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from unittest_reinvent.fixtures.paths import  ROCS_SHAPE_QUERY
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum, ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import METAMIZOLE
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityEnumerateStereo(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        specific_parameters = {rsp_enum.SHAPE_WEIGHT: 0.5,
                               rsp_enum.COLOR_WEIGHT: 0.5,
                               rsp_enum.SIM_MEASURE: sim_measure_enum.REF_TVERSKY,
                               rsp_enum.ROCS_INPUT: ROCS_SHAPE_QUERY,
                               rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
                               rsp_enum.ENUM_STEREO: False,
                               rsp_enum.MAX_STEREO: 3,
                               rsp_enum.MAX_CPUS: 8
                               }
        rocs_sim = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                        name="parallel_rocs_similarity",
                                        specific_parameters=specific_parameters)
        self.sf_state_no_enum = CustomSum(parameters=[rocs_sim])
        specific_parameters[rsp_enum.ENUM_STEREO] = True
        rocs_sim.specific_parameters = specific_parameters
        self.sf_state_enum = CustomSum(parameters=[rocs_sim])

    def test_parallel_rocs_similarity(self):
        smiles = [METAMIZOLE]
        score_no_enum = self.sf_state_no_enum.get_final_score(smiles=smiles)
        score_enum = self.sf_state_enum.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score_no_enum.total_score[0], 0.34, delta=0.01)
        self.assertAlmostEqual(score_enum.total_score[0], 0.34, delta=0.01)
