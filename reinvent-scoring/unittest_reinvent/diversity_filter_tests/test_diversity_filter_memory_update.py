import numpy as np

from reinvent_scoring.scoring.diversity_filters.curriculum_learning.memory_record_dto import MemoryRecordDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_diversity_filter_memory_base import BaseDiversityFilterMemory
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class TestDiversityFilterMemoryUpdate(BaseDiversityFilterMemory):

    def setUp(self):
        super().setUp()

        components = [
            ComponentSummary(np.array([21, 22, 23]), ComponentParameters("", "foo", 0)),
            ComponentSummary(np.array([31, 32, 33]), ComponentParameters("", "bar", 0)),
        ]

        index = 2  # Which element to extract from each ComponentSummary.
        self.total_score = 5
        update_dto = MemoryRecordDTO(index,self.step, self.total_score, self.smi, self.scaffold, '', components)
        self.mem.update(update_dto)
        self.df = self.mem.get_memory()

    def test_calling_update(self):
        self.assertEqual(self.df.loc[0, "foo"], 23)
        self.assertEqual(self.df.loc[0, "bar"], 33)
        self.assertEqual(self.df.loc[0, ScoringFunctionComponentNameEnum().TOTAL_SCORE], self.total_score)
        self.assertEqual(self.df.loc[0, "Step"], self.step)
        self.assertEqual(self.df.loc[0, "SMILES"], self.smi)
        self.assertEqual(self.df.loc[0, "Scaffold"], self.scaffold)