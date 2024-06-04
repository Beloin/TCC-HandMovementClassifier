import numpy as np
import antropy as ant
import nolds

import math


def rms(sample):
    return np.sqrt(np.mean(sample**2))


def waveformlen(sample):
    sum = 0
    for i in range(len(sample)-1):
        sum += math.fabs(sample[i+1] - sample[i])
    return sum


def wamp(sample, threshold=10):
    """
    @param: sample - Sample based on volts
    @param: threshold - Based on milivolts
    """
    threshold /= 1000
    sum = 0
    for i in range(len(sample)-1):
        v = math.fabs(sample[i] - sample[i+1])
        if v > threshold:
            sum += 1
    return sum


def app_entropy(sample):
    return ant.app_entropy(sample)

def sampen(sample):
    return nolds.sampen(sample)


def mav(sample):
    sum = 0
    for i in sample:
        sum += math.fabs(i)

    return sum/len(sample)
