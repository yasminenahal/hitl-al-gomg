import json
import os
import shutil
import unittest

import numpy.testing as npt
import pytest

from reinvent_chemistry.conversions import Conversions
from reinvent_scoring.scoring.score_components.structural.azdock import AZdock
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH, AZDOCK_UNITTEST_JSON, AZDOCK_UNITTEST_OE_RECEPTOR_PATH
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_AZdock_component_parameters
from unittest_reinvent.fixtures.test_data import PARACETAMOL, ASPIRIN, CAFFEINE
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from unittest_reinvent.unit_testing import ignore_warnings


class Test_structural_AZdock(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.csp_enum = ComponentSpecificParametersEnum()
        chemistry = Conversions()
        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)
        cls.old_dir = os.getcwd()
        os.chdir(MAIN_TEST_PATH)

        # load internal "AZdock" configuration file and update the receptor and ligand paths, respectively
        with open(AZDOCK_UNITTEST_JSON, 'r') as f:
            azdock_conf = json.load(f)
        azdock_conf["docking"]["docking_runs"][0]["parameters"]["receptor_paths"] = [AZDOCK_UNITTEST_OE_RECEPTOR_PATH]

        # write a temporary copy of the configuration file out
        temp_conf_path = MAIN_TEST_PATH + "/temp_azdock_unittest_OpenEye.json"
        with open(temp_conf_path, 'w') as f:
            json.dump(azdock_conf, f, indent=4)

        # initialize AZdock
        parameters = create_AZdock_component_parameters()
        parameters.specific_parameters[cls.csp_enum.AZDOCK_CONFPATH] = temp_conf_path
        cls.component = AZdock(parameters)
        cls.query_smiles = [ASPIRIN, PARACETAMOL, CAFFEINE]
        cls.query_mols = [chemistry.smile_to_mol(smile) for smile in cls.query_smiles]

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.old_dir)
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    @pytest.mark.integration
    @ignore_warnings
    def test_AZdock(self):
        summary = self.component.calculate_score(self.query_mols)
        npt.assert_almost_equal(summary.total_score, [0.13386323, 0.04236054, 0.03638716], decimal=3)
        self.assertEqual(len(summary.total_score), 3)
