import unittest

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ScoringFunctionNameEnum
from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters
from unittest_reinvent.fixtures.test_data import CELECOXIB
from unittest_reinvent.scoring_tests.fixtures import create_activity_component_regression, \
    create_predictive_property_component_regression


class TestParallelProduct(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sf_name_enum = ScoringFunctionNameEnum()
        predictive_property = create_predictive_property_component_regression()
        activity = create_activity_component_regression()
        qed_score = ComponentParameters(component_type=sf_enum.QED_SCORE,
                                        name="qed_score_name",
                                        weight=1.,
                                        specific_parameters={})

        sf_parameters = ScoringFunctionParameters(name=sf_name_enum.CUSTOM_PRODUCT,
                                                 parameters=[vars(activity), vars(qed_score),
                                                             vars(predictive_property)],
                                                 parallel=True)
        self.sf_instance = ScoringFunctionFactory(sf_parameters=sf_parameters)

    def test_parallel_1(self):
        smiles = [CELECOXIB] * 128
        score = self.sf_instance.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score[0], 0.258, 3)
