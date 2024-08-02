
import unittest
from unittest.mock import MagicMock, patch
import numpy.testing as npt

from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.score_components.pip import StringPiPPredictionComponent
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score
from unittest_reinvent.fixtures.test_data import CELECOXIB, BUTANE, ASPIRIN
from unittest_reinvent.scoring_tests.scoring_components.pip.utils import patch_pip_response


class Test_pip_HERG(unittest.TestCase):

    def setUp(self):
        enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        self.params = ComponentParameters(
            component_type=enum.THP1_CYTOTOXICITY,
            name="THP-1 Cytotoxicity class",
            weight=1.0,
            specific_parameters={
                csp_enum.VALUE_MAPPING: {
                    'inactive': 1.0,
                    'weak': 0.5,
                    'active': 0.0,
                }
            }
        )
        self.component = StringPiPPredictionComponent(self.params)
        self.query_smiles = [CELECOXIB, BUTANE, ASPIRIN]
        self.expected_raw_scores = ['active', 'weak', 'inactive']  # those classes may not be real, but since we're mocking it does not matter
        self.expected_scores = [0.0, 0.5, 1.0]

    def test_positive_case(self):
        with patch_pip_response(self.expected_raw_scores):
            total_score = score(self.component, self.query_smiles)
        npt.assert_almost_equal(total_score, self.expected_scores, decimal=3)
