import unittest
import pytest

import numpy.testing as npt

from reinvent_scoring import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY_3
from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from unittest_reinvent.fixtures.test_data import COCAINE, CAFFEINE, CYCLODECANE, PARACETAMOL
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestRocsSimilarityTverskyScoreBug(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        csp_enum = ComponentSpecificParametersEnum()
        specific_parameters = {"shape_weight": 0.0, "color_weight": 1.0,
                               "similarity_measure": sim_measure_enum.REF_TVERSKY,
                               "rocs_input": ROCS_SHAPE_QUERY_3,
                               "input_type": input_type_enum.SHAPE_QUERY,
                               csp_enum.TRANSFORMATION: {},
                               "max_num_cpus": 8
                               }
        rocs_sim = component_parameters(component_type=sf_enum.ROCS_SIMILARITY, specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[rocs_sim])

    def test_parallel_rocs_similarity(self):
        smiles = [COCAINE, CAFFEINE, CYCLODECANE, PARACETAMOL]
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_less(score.total_score, [1.0, 1.0, 1.0, 1.0])
