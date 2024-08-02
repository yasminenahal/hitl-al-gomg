import unittest
import pytest
import numpy.testing as npt

from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from unittest_reinvent.fixtures.paths import ROCS_HIGH_ENERGY_QRY
from reinvent_scoring.scoring.enums import ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import METAMIZOLE
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityOmegaopts(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        specific_parameters = {rsp_enum.SHAPE_WEIGHT: 0.5,
                               rsp_enum.COLOR_WEIGHT: 0.5,
                               rsp_enum.ROCS_INPUT: ROCS_HIGH_ENERGY_QRY,
                               rsp_enum.INPUT_TYPE: input_type_enum.SDF_QUERY,
                               rsp_enum.EWINDOW: 50,
                               rsp_enum.MAX_CONFS: 600,
                               rsp_enum.MAX_CPUS: 4
                               }
        ts_parameters = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                             name="parallel_rocs_similarity",
                                             specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_parallel_rocs_similarity_1(self):
        smiles = [METAMIZOLE]
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, [0.39], 2)
