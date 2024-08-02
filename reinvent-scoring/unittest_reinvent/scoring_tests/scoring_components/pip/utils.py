from unittest.mock import patch
from typing import List, Union


TARGET = 'reinvent_scoring.scoring.score_components.pip.base_pip_model_component.BasePiPModelComponent._execute_request'


def patch_pip_response(expected_scores: List[Union[float, str]], predicion_key: str = 'prediction'):
    response = {
        'jsonData': {
            'data': [
                {'id': i, predicion_key: p} for i, p in enumerate(expected_scores)
            ]
        }
    }

    return patch(TARGET, return_value=response)


def patch_pip_log_response(expected_scores: List[Union[float, str]]):
    return patch_pip_response(expected_scores, predicion_key='log_prediction')
