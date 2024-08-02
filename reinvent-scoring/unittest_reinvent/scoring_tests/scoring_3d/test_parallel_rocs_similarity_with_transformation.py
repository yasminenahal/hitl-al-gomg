import unittest
import pytest

from reinvent_scoring.scoring import CustomSum
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum
from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY
from reinvent_scoring.scoring.enums import ROCSSimilarityMeasuresEnum, ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE
from unittest_reinvent.scoring_tests.scoring_3d.fixtures import component_parameters


@pytest.mark.integration
class TestParallelRocsSimilarityWithTransformation(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        csp_enum = ComponentSpecificParametersEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        tt_enum = TransformationTypeEnum()
        specific_parameters = {
            rsp_enum.SHAPE_WEIGHT: 0.5,
            rsp_enum.COLOR_WEIGHT: 0.5,
            rsp_enum.SIM_MEASURE: sim_measure_enum.REF_TVERSKY,
            rsp_enum.ROCS_INPUT: ROCS_SHAPE_QUERY,
            rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
            csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.LOW: 0.3,
                TransformationParametersEnum.HIGH: 0.7,
                TransformationParametersEnum.K: 1,
                TransformationParametersEnum.TRANSFORMATION_TYPE: tt_enum.REVERSE_SIGMOID
            }
        }
        ts_parameters = component_parameters(component_type=sf_enum.PARALLEL_ROCS_SIMILARITY,
                                             name="parallel_rocs_similarity",
                                             specific_parameters=specific_parameters)
        self.sf_state = CustomSum(parameters=[ts_parameters])

    def test_rocs_similarity_1(self):
        smiles = [CELECOXIB]
        score = self.sf_state.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score, [1.0], delta=0.01)

    def test_rocs_similarity_2(self):
        smiles = [METAMIZOLE]
        score = self.sf_state.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score, [1.0], delta=0.01)
