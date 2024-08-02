import unittest
import pytest

from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY
from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestRocsSimilarityWithTransformation(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        csp_enum = ComponentSpecificParametersEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        tt_enum = TransformationTypeEnum()
        specific_parameters = {
            "shape_weight": 0.5, "color_weight": 0.5,
            "similarity_measure": sim_measure_enum.REF_TVERSKY,
            "rocs_input": ROCS_SHAPE_QUERY,
            "input_type": input_type_enum.SHAPE_QUERY,
            csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.LOW: 0.3,
                TransformationParametersEnum.HIGH: 0.7,
                TransformationParametersEnum.K: 1,
                TransformationParametersEnum.TRANSFORMATION_TYPE: tt_enum.REVERSE_SIGMOID
            }
        }
        ts_parameters = component_parameters(component_type=sf_enum.ROCS_SIMILARITY,
                                             specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_rocs_similarity_1(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score, [1.0], delta=0.01)

    def test_rocs_similarity_2(self):
        score = self.sf_state.get_final_score(smiles=[METAMIZOLE])
        self.assertAlmostEqual(score.total_score, [1.0], delta=0.01)
