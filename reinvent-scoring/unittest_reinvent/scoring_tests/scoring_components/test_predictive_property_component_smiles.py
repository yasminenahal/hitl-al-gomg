from typing import Union, List

import numpy.testing as npt

from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single, score
from unittest_reinvent.fixtures.test_data import CELECOXIB, BUTANE, PENTANE
from unittest_reinvent.scoring_tests.scoring_components.base_predictive_property_component import \
    BaseTestPredictivePropertyComponent


class ModelWithPredictFromSmiles:

    @staticmethod
    def predict_from_smiles(smiles: Union[str, List[str]]) -> List[float]:
        input = [smiles] if isinstance(smiles, str) else smiles  # Wrap single SMILES in a list.
        output = [len(smi) for smi in input]
        return output


class TestPredictivePropertyComponentWithPredictFromSmiles(BaseTestPredictivePropertyComponent):

    @classmethod
    def setUpClass(cls):
        cls.model = ModelWithPredictFromSmiles()
        super().setUpClass()

    def test_predictive_property_1(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB), 50, 3)

    def test_predictive_property_2(self):
        npt.assert_almost_equal(score(self.component, [BUTANE, PENTANE]), [4, 5], 3)
