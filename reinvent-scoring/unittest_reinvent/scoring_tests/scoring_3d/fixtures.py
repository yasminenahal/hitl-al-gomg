from typing import Dict, Any
from reinvent_scoring.scoring.component_parameters import ComponentParameters


def component_parameters(component_type: str, specific_parameters: Dict[str, Any], name: str = "rocs_similarity", ) \
        -> ComponentParameters:
    return ComponentParameters(component_type=component_type,
                               name=name,
                               weight=1.,
                               specific_parameters=specific_parameters)
