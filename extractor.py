from typing import Any
import numpy as np


class Cycle:
    rest = []
    extension = []
    flexion = []
    ulnar_deviation = []
    radial_deviation = []
    grip = []
    finger_abduction = []
    finger_adduction = []
    supination = []
    pronation = []


    def __str__(self) -> str:
        return f"Cycle Object with {len(self.rest)} lenght"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __getitem__(self, _name):
        return getattr(self, _name)

    def __setitem__(self, _name, _value):
        setattr(self, _name, _value)

class RecordExtractor(object):

    _freq = None
    _norm = False

    def __init__(self, freq=2000, norm=False):
        self._freq = freq
        self._norm = norm

    def read_sample(self, samples):
        """ 
        samples should be a Dataframe with 4 columns as 4 Sensors were available
        Returns an array with 4 elements being another array with 5 elements, those 5 elements
        are the cycles

        @return An array shaped as (4, 5), 4 Sensors and 5 cycles each
        """
        cycles = []
        for i in range(4):
            cycles.append([])
            for j in range(5):
                nc = self.read_cycle(j, samples.iloc[:, i])
                cycles[i].append(nc)

        return cycles

    # Cycle starts with 0
    def read_cycle(self, n, sample):
        offset = n*(104 * self._freq) + n*(30 * self._freq)
        offsample = sample[offset:]

        c = Cycle()

        # 0°
        # c.rest = offsample[self.sec(4):self.sec(6)]
        c.rest = self.read_nthc(0, offsample)
        c.extension = self.read_nthc(1, offsample)
        c.flexion = self.read_nthc(2, offsample)
        c.ulnar_deviation = self.read_nthc(3, offsample)
        c.radial_deviation = self.read_nthc(4, offsample)
        c.grip = self.read_nthc(5, offsample)
        c.finger_abduction = self.read_nthc(6, offsample)
        c.finger_adduction = self.read_nthc(7, offsample)
        c.supination = self.read_nthc(8, offsample)
        c.pronation = self.read_nthc(9, offsample)

        return c

    def read_nthc(self, n, sample):
        offset = self.to_sec(4) + n*(self.to_sec(6) + self.to_sec(4))
        r = sample[offset:offset + self.to_sec(6)]

        if self._norm:
            return np.array(r)

        return r

    def to_sec(self, s):
        return s * self._freq
    
    def to_ms(self, ms):
        return (ms/1000) * self._freq
