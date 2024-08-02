import unittest, pytest

import numpy.testing as npt
from reinvent_chemistry.conversions import Conversions

from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.score_components.synthetic_accessibility.sas_component import SASComponent
from unittest_reinvent.fixtures.paths import SAS_MODEL_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, ANILINE

@pytest.mark.integration
class Test_sas_component(unittest.TestCase):

    def setUp(self):
        csp_enum = ComponentSpecificParametersEnum()
        ts_parameters = create_activity_component_regression()
        ts_parameters.specific_parameters[csp_enum.TRANSFORMATION] = {}
        # ts_parameters.specific_parameters[csp_enum.MODEL_PATH] = SAS_MODEL_PATH
        #FIXME: currently the property above isnt used
        ts_parameters.specific_parameters["saz_model_path"] = SAS_MODEL_PATH
        chemistry = Conversions()

        self.query_smiles = [CELECOXIB, ASPIRIN, ANILINE]
        self.query_mols = [chemistry.smile_to_mol(smile) for smile in self.query_smiles]
        self.component = SASComponent(ts_parameters)

    def test_sas(self):
        summary = self.component.calculate_score(self.query_mols)
        npt.assert_almost_equal(summary.total_score, [0.97, 0.97, 0.98], decimal=2)
