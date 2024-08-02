import unittest
import pytest

from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.paths import ROCS_MULTI_SIMILARITY_TEST_DATA
from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE, INVALID
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsMultiSimilarity(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        csp_enum = ComponentSpecificParametersEnum()
        # NOTE: license is required
        # "export OE_LICENSE='/opt/scp/software/oelicense/1.0/oe_license.seq1'"
        specific_parameters = {"shape_weight": 0.5, "color_weight": 0.5,
                               "rocs_input": ROCS_MULTI_SIMILARITY_TEST_DATA,
                               "input_type": input_type_enum.SDF_QUERY,
                               csp_enum.TRANSFORMATION: {},
                               "max_num_cpus": 8
                               }
        ts_parameters = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                             name="rocs_multi_similarity",
                                             specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_rocs_similarity_1(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertGreater(score.total_score, [0.94])

    def test_rocs_similarity_2(self):
        score = self.sf_state.get_final_score(smiles=[METAMIZOLE])
        self.assertAlmostEqual(score.total_score, [0.39], delta=0.01)

    def test_rocs_similarity_3(self):
        score = self.sf_state.get_final_score(smiles=[INVALID])
        self.assertEqual(score.total_score, [0.0])
