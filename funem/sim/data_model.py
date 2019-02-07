import numpy as np

from utils.common_defs import D1AMP, D2AMP, D3AMP, D4AMP
from utils.common_defs import D1FRQ, D2FRQ, D3FRQ, D4FRQ
from utils.common_defs import N1AMP, N2AMP, N3AMP, N4AMP
from utils.common_defs import DTYPE


class DataGenerator(object):
    '''
    generate a set of oscillation patterns of different frequencies as function
    of a discrete time-step - user must manually `step` the system with time.
    '''
    amp = np.asarray([D1AMP, D2AMP, D3AMP, D4AMP], dtype=DTYPE)
    frq = np.pi * np.asarray([D1FRQ, D2FRQ, D3FRQ, D4FRQ], dtype=DTYPE)

    def __init__(self, time_step=0.01):
        self.t = 0
        self.time_step = time_step
        assert len(DataGenerator.amp) == len(DataGenerator.frq)

    def _gen_point(self):
        raw_vs = []
        for i in range(len(DataGenerator.amp)):
            raw_vs.append(
                DataGenerator.amp[i] * np.cos(DataGenerator.frq[i] * self.t)
            )
        return np.asarray(raw_vs, dtype=DTYPE)

    def step(self):
        self.t = self.t + self.time_step
        data = self._gen_point()
        return data


class NoiseModel(object):
    '''
    generate a set of noise complimentary to the 'true' values from a
    `DataGenerator` object. noise is a function of the true data values and
    not a function of time.
    '''
    default_noise_scale = [N1AMP, N2AMP, N3AMP, N4AMP]

    def __init__(self, random_seed=None, drop_probability=0.0, noise_array=None):
        self.noise = noise_array or np.asarray(
            NoiseModel.default_noise_scale, dtype=DTYPE
        )
        assert len(self.noise) == 4
        if random_seed is not None:
            np.random.seed(random_seed)

    def gen_noise(self, data):
        assert len(data) == 4
        noise_values = []
        for i, d in enumerate(data):
            noise_values.append(
                self.noise[i] * data[i] * np.random.randn()
            )
        return np.asarray(noise_values, dtype=DTYPE)
