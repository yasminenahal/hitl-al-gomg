import unittest

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ScoringFunctionNameEnum
from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import \
    create_predictive_property_component_regression


class TestParallelAdditive(unittest.TestCase):

    def setUp(self):
        component_enum = ScoringFunctionComponentNameEnum()
        sf_enum = ScoringFunctionNameEnum()
        ts_parameters2 = create_predictive_property_component_regression()
        ts_parameters2.weight = 1

        ts_parameters = ComponentParameters(component_type=component_enum.TANIMOTO_SIMILARITY,
                                            name="tanimoto_similarity",
                                            weight=1.,
                                            specific_parameters={"smiles":[ASPIRIN]})

        sf_parameters = ScoringFunctionParameters(name=sf_enum.CUSTOM_SUM,
                                                 parameters=[vars(ts_parameters), vars(ts_parameters2)],
                                                 parallel=True)
        self.sf_instance = ScoringFunctionFactory(sf_parameters=sf_parameters)

    def test_parallel_rocs_similarity_1(self):
        smiles = [CELECOXIB]*128
        score = self.sf_instance.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score[0],  0.148, 3)

