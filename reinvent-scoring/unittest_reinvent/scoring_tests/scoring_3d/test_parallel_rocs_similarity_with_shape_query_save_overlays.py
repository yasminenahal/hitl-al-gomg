import unittest
import pytest
import os
import shutil

import numpy.testing as npt
from rdkit.Chem import SDMolSupplier

from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum, ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityWithShapeQuerySaveOverlays(unittest.TestCase):

    def setUp(self):
        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        overlays_dir = os.path.join(MAIN_TEST_PATH, "test_save_rocs")
        prefix = "unit_test_"
        specific_parameters = {rsp_enum.SHAPE_WEIGHT: 0.5,
                               rsp_enum.COLOR_WEIGHT: 0.5,
                               rsp_enum.SIM_MEASURE: sim_measure_enum.REF_TVERSKY,
                               rsp_enum.ROCS_INPUT: ROCS_SHAPE_QUERY,
                               rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
                               rsp_enum.SAVE_ROCS_OVERLAYS: True,
                               rsp_enum.ROCS_OVERLAYS_DIR: overlays_dir,
                               rsp_enum.ROCS_OVERLAYS_PREFIX: prefix,
                               rsp_enum.MAX_CPUS: 8
                               }
        rocs_sim = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                        name="parallel_rocs_similarity",
                                        specific_parameters=specific_parameters)
        self.file_name = os.path.join(overlays_dir, prefix + "-001.sdf")
        self.sf_state = CustomSum(parameters=[rocs_sim])

    def tearDown(self):
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_parallel_rocs_similarity(self):
        smiles = [CELECOXIB]*128
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score[0], [0.38], 2)

        file_created = os.path.exists(self.file_name) and os.path.getsize(self.file_name) > 0
        self.assertTrue(file_created)
        num_mols = -1
        if file_created:
            suppl = SDMolSupplier(self.file_name)
            num_mols = len(suppl)
        self.assertEqual(num_mols, len(score.total_score))
