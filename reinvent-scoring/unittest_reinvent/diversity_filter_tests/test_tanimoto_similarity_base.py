import os
import shutil
import unittest
import numpy as np

from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from unittest_reinvent.fixtures.test_data import ASPIRIN, PROPANE, ETHANE, PENTANE
from reinvent_scoring.scoring.enums.diversity_filter_enum import DiversityFilterEnum

from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.score_summary import FinalSummary, ComponentSummary
from reinvent_scoring.scoring.component_parameters import ComponentParameters


class BaseTanimotoSimilarity(unittest.TestCase):

    def setUp(self):
        self.scaffold_enum = DiversityFilterEnum()
        self.sf_enum = ScoringFunctionComponentNameEnum()
        self.workfolder = MAIN_TEST_PATH
        # create a scaffold filter and fill it with a few entries
        smiles = [ASPIRIN, PROPANE, PENTANE, ETHANE]
        scores = np.array([1.0, 0.5, 0.4, 0.3])
        valid_idx = [0, 1, 2, 3]
        component_parameters = ComponentParameters(component_type=self.sf_enum.TANIMOTO_SIMILARITY,
                                                   name="tanimoto_similarity",
                                                   weight=1.,
                                                   specific_parameters={})
        component_score_summary = ComponentSummary(scores, component_parameters)

        final_summary = FinalSummary(scores, smiles, valid_idx, [component_score_summary])
        update_dto = UpdateDiversityFilterDTO(final_summary, [])

        sf_parameters = DiversityFilterParameters(name=self.scaffold_enum.NO_FILTER, minscore=0.5, minsimilarity=0.4,
                                                  bucket_size=1)

        self.scaffold_filter = DiversityFilter(sf_parameters)
        self.scaffold_filter.update_score(update_dto)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)
