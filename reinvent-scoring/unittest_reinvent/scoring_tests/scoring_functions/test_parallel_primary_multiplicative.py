import unittest

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ScoringFunctionNameEnum
from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters
from unittest_reinvent.fixtures.test_data import BENZENE, CELECOXIB, ETHANE
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import \
    create_predictive_property_component_regression


class TestParallelPrimaryMultiplicativeFunction(unittest.TestCase):

    def setUp(self):
        enum = ScoringFunctionComponentNameEnum()
        sf_name_enum = ScoringFunctionNameEnum()
        activity = create_predictive_property_component_regression()
        activity.weight = 1
        qed_score = ComponentParameters(component_type=enum.QED_SCORE,
                                        name="qed_score_name",
                                        weight=1.,
                                        specific_parameters={})
        custom_alerts = ComponentParameters(component_type=enum.CUSTOM_ALERTS,
                                            name="custom_alerts_name",
                                            weight=1.,
                                            specific_parameters={"smiles":[BENZENE]})
        matching_substructure = ComponentParameters(component_type=enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure_name",
                                                    weight=1.,
                                                    specific_parameters={})

        sf_parameters = ScoringFunctionParameters(name=sf_name_enum.CUSTOM_PRODUCT,
                                                 parameters=[vars(activity), vars(qed_score), vars(custom_alerts),
                                                             vars(matching_substructure)], parallel=True)
        self.sf_instance = ScoringFunctionFactory(sf_parameters=sf_parameters)

    def test_primary_multiplicative_1(self):
        smiles = [CELECOXIB]*3
        score = self.sf_instance.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score[0], 0)

    def test_primary_multiplicative_2(self):
        smiles = [ETHANE]*3
        score = self.sf_instance.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score[0], 0.237, 3)
