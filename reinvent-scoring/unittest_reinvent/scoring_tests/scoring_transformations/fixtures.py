from typing import List

def round_list(lst: List[float], decimals: int) -> List[float]:
    return list(map(lambda x: round(x, decimals), lst))
