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
from reinvent_scoring.scoring.score_components.pip.pip_prediction_component import PiPPredictionComponent
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_c_lab_component
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score
from unittest_reinvent.fixtures.test_data import CELECOXIB, BUTANE, ASPIRIN
from unittest_reinvent.scoring_tests.scoring_components.pip.utils import patch_pip_response


class Test_pip_HERG(unittest.TestCase):

    def setUp(cls):
        enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        parameters = create_c_lab_component(enum.HERG_PIP)
        parameters.specific_parameters[csp_enum.TRANSFORMATION] = {}
        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)

        cls.query_smiles = [CELECOXIB, BUTANE, ASPIRIN]
        cls.expected_scores = [10.969, 53.071, 100.129]

        cls.component = PiPPredictionComponent(parameters)

    def tearDown(cls):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_pip_1(self):
        with patch_pip_response(self.expected_scores):
            npt.assert_almost_equal(score(self.component, self.query_smiles), self.expected_scores, decimal=3)

    def test_pip_empty_response(self):
        with patch_pip_response([]):
            npt.assert_almost_equal(score(self.component, self.query_smiles), [0, 0, 0], 3)


class Test_pip_HERG_transformation(unittest.TestCase):
    def setUp(cls):
        csp_enum = ComponentSpecificParametersEnum()
        transf_type = TransformationTypeEnum()
        enum = ScoringFunctionComponentNameEnum()
        ts_parameters = create_c_lab_component(enum.HERG_PIP)
        ts_parameters.specific_parameters[csp_enum.TRANSFORMATION].update({
            TransformationParametersEnum.TRANSFORMATION_TYPE: transf_type.SIGMOID,
            TransformationParametersEnum.HIGH: 100,
            TransformationParametersEnum.LOW: 10,
            TransformationParametersEnum.K: 0.5,
        })

        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)

        cls.query_smiles = [CELECOXIB, BUTANE, ASPIRIN]
        cls.expected_scores = [0.001, 0.001, 0.001]
        cls.component = PiPPredictionComponent(ts_parameters)

    def tearDown(self):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_pip_transformed_1(self):
        with patch_pip_response(self.expected_scores):
            npt.assert_almost_equal(score(self.component, self.query_smiles), self.expected_scores, decimal=3)

    def test_pip_empty_response(self):
        with patch_pip_response([]):
            npt.assert_almost_equal(score(self.component, self.query_smiles), [0, 0, 0], 3)
