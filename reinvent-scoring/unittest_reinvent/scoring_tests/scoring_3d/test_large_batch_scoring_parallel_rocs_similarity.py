import unittest
import pytest

import numpy.testing as npt

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring import CustomProduct
from unittest_reinvent.fixtures.paths import ROCS_SHAPE_QUERY_BATCH
from reinvent_scoring.scoring.enums import ROCSInputFileTypesEnum, ROCSSimilarityMeasuresEnum, \
    ROCSSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE, AMOXAPINE, METHOXYHYDRAZINE, COCAINE, \
    CAFFEINE, CYCLODECANE, PARACETAMOL, ASPIRIN


@pytest.mark.integration
class TestParallelRocsSimilarityWithShapeQueryLargeBatch(unittest.TestCase):
    # This is to assert that there is always a 1:1 between smiles and scores for each batch
    # even when one or more smiles fail to produce a score
    def setUp(self):
        enum = ScoringFunctionComponentNameEnum()
        sim_measure_enum = ROCSSimilarityMeasuresEnum()
        input_type_enum = ROCSInputFileTypesEnum()
        csp_enum = ComponentSpecificParametersEnum()
        rsp_enum = ROCSSpecificParametersEnum()
        custom_alerts = ComponentParameters(component_type=enum.CUSTOM_ALERTS,
                                            name="custom_alerts_name",
                                            weight=1.,
                                            specific_parameters={})
        matching_substructure = ComponentParameters(component_type=enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure_name",
                                                    weight=1.,
                                                    specific_parameters={})
        specific_parameters = {rsp_enum.SHAPE_WEIGHT: 0.5,
                               rsp_enum.COLOR_WEIGHT: 0.5,
                               rsp_enum.SIM_MEASURE: sim_measure_enum.REF_TVERSKY,
                               rsp_enum.ROCS_INPUT: ROCS_SHAPE_QUERY_BATCH,
                               rsp_enum.INPUT_TYPE: input_type_enum.SHAPE_QUERY,
                               csp_enum.TRANSFORMATION: {},
                               rsp_enum.MAX_CPUS: 4
                               }
        rocs_sim = ComponentParameters(component_type=enum.PARALLEL_ROCS_SIMILARITY,
                                       name="parallel_rocs_similarity",
                                       weight=3.,
                                       specific_parameters=specific_parameters)
        self.sf_state = CustomProduct(parameters=[custom_alerts, matching_substructure, rocs_sim])

    def test_quick(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB, PARACETAMOL, METAMIZOLE])
        vals = [0.63, 0.34, 0.58]
        npt.assert_array_almost_equal(score.total_score, vals, 2)

    def test_extended(self):
        score = self.sf_state.get_final_score(
            smiles=[AMOXAPINE, METHOXYHYDRAZINE, COCAINE, CAFFEINE, CYCLODECANE, PARACETAMOL, ASPIRIN]
        )
        vals = [0.36, 0.04, 0.52, 0.38, 0.08, 0.34, 0.54]
        npt.assert_array_almost_equal(score.total_score, vals, 2)
