import os
import shutil
import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest

from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import TransformationParametersEnum
from reinvent_scoring.scoring.score_components.pip.pip_prediction_component import PiPPredictionComponent
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_c_lab_component
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score
from unittest_reinvent.fixtures.test_data import CELECOXIB, BUTANE, PENTANE
from unittest_reinvent.scoring_tests.scoring_components.pip.utils import patch_pip_response


class Test_pip_caco2_efflux(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        parameters = create_c_lab_component(enum.CACO2_EFFLUX_PIP)
        parameters.specific_parameters[csp_enum.TRANSFORMATION] = {}
        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)

        cls.query_smiles = [CELECOXIB, BUTANE, PENTANE]
        cls.expected_scores = [0.7, 2.2, 2.1]

        cls.component = PiPPredictionComponent(parameters)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_pip_1(self):
        with patch_pip_response(self.expected_scores):
            npt.assert_almost_equal(score(self.component, self.query_smiles), self.expected_scores, decimal=1)

    def test_pip_empty_response(self):
        with patch_pip_response([]):
            npt.assert_almost_equal(score(self.component, self.query_smiles), [0, 0, 0], 3)


class Test_pip_caco2_efflux_transformation(unittest.TestCase):
    def setUp(cls):
        enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        parameters = create_c_lab_component(enum.CACO2_EFFLUX_PIP)
        parameters.specific_parameters[csp_enum.TRANSFORMATION].update({
            TransformationParametersEnum.HIGH: 3,
            TransformationParametersEnum.LOW: 0,
        })

        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)

        cls.query_smiles = [CELECOXIB, BUTANE, PENTANE]
        cls.expected_raw_scores = [0.795, 1.587, 1.55]
        cls.expected_scores = [0.17, 0.19, 0.19]
        cls.component = PiPPredictionComponent(parameters)

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_clab_transformed_1(self):
        with patch_pip_response(self.expected_raw_scores):
            result = score(self.component, self.query_smiles)
        has_null = any(np.isnan(result))
        npt.assert_(not has_null, "returned array has Null values")
        unique = np.unique(result)
        npt.assert_equal(np.sort(unique), np.sort(result))
