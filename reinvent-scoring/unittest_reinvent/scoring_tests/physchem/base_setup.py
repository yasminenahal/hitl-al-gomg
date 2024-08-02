import unittest

from reinvent_scoring import ComponentSpecificParametersEnum, ScoringFunctionComponentNameEnum, TransformationTypeEnum
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring import CustomSum


class BaseSetup(unittest.TestCase):

    def init(self, component_type, specific_parameters):
        self.component_type = component_type
        self.specific_parameters = specific_parameters
        
    def setup_attrs(self):
        self.csp_enum = ComponentSpecificParametersEnum()
        self.tt_enum = TransformationTypeEnum()
        self.sf_enum = ScoringFunctionComponentNameEnum()

    def setUp(self):
        ts_parameters = ComponentParameters(component_type=self.component_type,
                                            name="dummy",
                                            weight=1.,
                                            specific_parameters=self.specific_parameters)
        self.sf_state = CustomSum(parameters=[ts_parameters])
