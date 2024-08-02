import os
import shutil
import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest

from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum
from reinvent_scoring.scoring.score_components.pip.pip_log_prediction_component import PiPLogPredictionComponent
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_c_lab_component
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score
from unittest_reinvent.fixtures.test_data import CELECOXIB, BUTANE, PENTANE
from unittest_reinvent.scoring_tests.scoring_components.pip.utils import patch_pip_log_response


class Test_pip_hlm_clint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        parameters = create_c_lab_component(enum.HLM_CLINT_PIP)
        parameters.specific_parameters[csp_enum.TRANSFORMATION] = {}
        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)

        cls.query_smiles = [CELECOXIB, BUTANE, PENTANE]
        cls.expected_scores = [1.201, 0.906,  1.087]

        cls.component = PiPLogPredictionComponent(parameters)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_pip_1(self):
        with patch_pip_log_response(self.expected_scores):
            npt.assert_almost_equal(score(self.component, self.query_smiles), self.expected_scores, decimal=1)

    def test_pip_empty_response(self):
        with patch_pip_log_response([]):
            values = score(self.component, self.query_smiles)
        npt.assert_almost_equal(values, [0, 0, 0], 3)


class Test_pip_hlm_clint_transformation(unittest.TestCase):
    def setUp(cls):
        csp_enum = ComponentSpecificParametersEnum()
        transf_type = TransformationTypeEnum()
        enum = ScoringFunctionComponentNameEnum()
        parameters = create_c_lab_component(enum.HLM_CLINT_PIP)
        parameters.specific_parameters[csp_enum.TRANSFORMATION].update({
            TransformationParametersEnum.TRANSFORMATION_TYPE: transf_type.STEP,
            TransformationParametersEnum.LOW: 1.0
        })
        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)

        cls.query_smiles = [CELECOXIB, BUTANE, PENTANE]
        cls.expected_raw_scores = [1.2, 0.9, 1.1]
        cls.expected_scores = [1.0, 0.0, 1.0]
        cls.component = PiPLogPredictionComponent(parameters)

    def tearDown(self):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_pip_transformed_1(self):
        with patch_pip_log_response(self.expected_raw_scores):
            npt.assert_almost_equal(score(self.component, self.query_smiles), self.expected_scores, decimal=3)

    def test_pip_empty_response(self):
        with patch_pip_log_response([]):
            npt.assert_almost_equal(score(self.component, self.query_smiles), [0, 0, 0], 3)
