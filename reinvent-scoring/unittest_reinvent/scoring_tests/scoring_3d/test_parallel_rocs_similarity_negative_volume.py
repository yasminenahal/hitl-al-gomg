import unittest
import pytest
import numpy.testing as npt

from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from unittest_reinvent.fixtures.paths import ROCS_CUSTOM_CFF, ROCS_NEG_VOL_SQ, ROCS_NEG_VOL_LIG, \
    ROCS_NEG_VOL_PROTEIN
from reinvent_scoring.scoring.enums import ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import AMOXAPINE, METHOXYHYDRAZINE, PARACETAMOL
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityNegativeVolume(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        specific_parameters = {rsp_enum.SHAPE_WEIGHT: 0.5,
                               rsp_enum.COLOR_WEIGHT: 0.5,
                               rsp_enum.ROCS_INPUT: ROCS_NEG_VOL_SQ,
                               rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
                               rsp_enum.CUSTOM_CFF: ROCS_CUSTOM_CFF,
                               rsp_enum.NEGATIVE_VOLUME: True,
                               rsp_enum.PROTEIN_NEG_VOL_FILE: ROCS_NEG_VOL_PROTEIN,
                               rsp_enum.LIGAND_NEG_VOL_FILE: ROCS_NEG_VOL_LIG,
                               rsp_enum.ENUM_STEREO: True,
                               rsp_enum.MAX_STEREO: 3,
                               rsp_enum.MAX_CPUS: 8
                               }
        ts_parameters = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                             name="parallel_rocs_similarity",
                                             specific_parameters=specific_parameters)
        self.sf_state_with_vol = CustomSum(parameters=[ts_parameters])
        specific_parameters[rsp_enum.NEGATIVE_VOLUME] = False
        ts_parameters.specific_parameters = specific_parameters
        self.sf_state_no_vol = CustomSum(parameters=[ts_parameters])

    def test_parallel_rocs_similarity_1(self):
        smiles = [AMOXAPINE, PARACETAMOL, METHOXYHYDRAZINE]
        score = self.sf_state_with_vol.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, [0.2, 0.13, 0.03], 2)

    def test_parallel_rocs_similarity_2(self):
        smiles = [AMOXAPINE, PARACETAMOL, METHOXYHYDRAZINE]
        score = self.sf_state_no_vol.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, [0.24, 0.17, 0.03], 2)
