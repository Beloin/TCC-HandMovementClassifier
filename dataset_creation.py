from typing import Tuple
import extractor as ext
import numpy as np
import pandas as pd
import feature_extractors as fe
import onset_extractor as osext

a = 12

def create_dataset(fbase: pd.DataFrame, t1=.35, t2=5, time_ms=299):
    """
    Parameters
    ----------
    fbase : Dataframe
        Current database extracted from csv file whitout headers
    """
    extractor = ext.RecordExtractor()
    cycles = extractor.read_sample(fbase)
    normal, offsetted =  cycle_through(cycles, extractor,  t1, t2, time_ms)

    return pd.DataFrame(normal), pd.DataFrame(offsetted)

# TODO: Add more limit
def cycle_through(cycles, extractor: ext.RecordExtractor,  t1=.35, t2=5, time_ms=299) -> Tuple[dict, dict]:
    classes = ["rest","extension", "flexion","ulnar_deviation","radial_deviation","grip","finger_abduction","finger_adduction","supination","pronation",]
    newds = {}
    newds_offsetted = {}
    newds["class"] = []
    newds_offsetted["class"] = []

    for j in range(4):
        for i in range(5):
            # lower_limit = int(extractor.to_ms(100))
            for c in classes:
                upper_limit = int(extractor.to_ms(time_ms))

                populatedict(cycles[j][i][c][:upper_limit], "SENSOR" + str(j), newds)

                if c == 'rest':
                    lower_limit = 0
                else:
                    lower_limit, _ = osext.extract_per_offset(cycles[j][i][c], t1,t2,ret_data=False)

                populatedict(cycles[j][i][c][lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

    for c in classes:
        for i in range(5):
            newds["class"].append(c)
            newds_offsetted["class"].append(c)

    return newds, newds_offsetted


def populatedict(data, name, dic:dict):
    data = np.array(data)
    if not dic.get(name+"_RMS"): dic[name+"_RMS"] = []
    if not dic.get(name+"_WAVELEN"): dic[name+"_WAVELEN"] = []
    if not dic.get(name+"_WAMP"): dic[name+"_WAMP"] = []
    if not dic.get(name+"_APPEN"): dic[name+"_APPEN"] = []
    if not dic.get(name+"_SAMPEN"): dic[name+"_SAMPEN"] = []
    if not dic.get(name+"_MAV"): dic[name+"_MAV"] = []

    dic[name+"_RMS"].append(fe.rms(data))
    dic[name+"_WAVELEN"].append(fe.waveformlen(data))
    dic[name+"_WAMP"].append(fe.wamp(data))
    dic[name+"_APPEN"].append(fe.app_entropy(data))
    dic[name+"_SAMPEN"].append(fe.sampen(data))
    dic[name+"_MAV"].append(fe.mav(data))