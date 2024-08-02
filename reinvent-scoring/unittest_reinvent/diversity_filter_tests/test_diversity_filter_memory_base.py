import unittest

from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterMemory
from unittest_reinvent.fixtures.test_data import TOLUENE, ANILINE, BENZENE


class BaseDiversityFilterMemory(unittest.TestCase):

    def setUp(self):
        self.mem = DiversityFilterMemory()
        self.smi = TOLUENE
        self.smi2 = ANILINE
        self.scaffold = BENZENE
        self.step = 10
