
from typing import List
import math


def extract_per_offset(data: List[float], threshold:float, timethreshold:int, ret_data=True):
    cum = 0
    start_index = 0
    for i, d in enumerate(data):
        if d >= math.fabs(threshold):
            cum +=1
            if cum >= timethreshold:
                start_index = i
                break

    return start_index, data[start_index:] if ret_data else None
