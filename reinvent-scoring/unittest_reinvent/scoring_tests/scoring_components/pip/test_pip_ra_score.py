import os
import shutil
import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum
from reinvent_scoring.scoring.score_components.pip.pip_prediction_component import PiPPredictionComponent
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_c_lab_component
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score
from unittest_reinvent.fixtures.test_data import CELECOXIB, ANILINE
from unittest_reinvent.scoring_tests.scoring_components.pip.utils import patch_pip_response


class Test_pip_RAScore(unittest.TestCase):
    def setUp(cls):
        csp_enum = ComponentSpecificParametersEnum()
        transf_type = TransformationTypeEnum()
        enum = ScoringFunctionComponentNameEnum()
        ts_parameters = create_c_lab_component(enum.RA_SCORE)
        ts_parameters.specific_parameters[csp_enum.TRANSFORMATION][
            TransformationParametersEnum.TRANSFORMATION_TYPE] = transf_type.NO_TRANSFORMATION

        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)

        cls.query_smiles = [CELECOXIB, ANILINE]
        cls.component = PiPPredictionComponent(ts_parameters)

    def tearDown(self):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_pip_transformed_1(self):
        with patch_pip_response([1.0, 0.0]):
            result = score(self.component, self.query_smiles)
        has_null = any(np.isnan(result))
        npt.assert_(not has_null, "returned array has Null values")
        unique = np.unique(result)
        npt.assert_equal(np.sort(unique), np.sort(result))

    def test_pip_empty_response(self):
        with patch_pip_response([]):
            npt.assert_almost_equal(score(self.component, self.query_smiles), [0, 0], 1)

