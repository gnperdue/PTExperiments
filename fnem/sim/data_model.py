import numpy as np


# data amplitudes
D1AMP, D2AMP, D3AMP, D4AMP = 10.0, 1.0, 0.5, 0.1
# data update frequencies / pi
D1FRQ, D2FRQ, D3FRQ, D4FRQ = 1.0, 0.1, 3.0, 10.0
DTYPE = np.float32
# noise amplitudes
N1AMP, N2AMP, N3AMP, N4AMP = 0.05, 0.04, 0.02, 0.01


class DataGenerator(object):
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
        data = self._gen_point()
        self.t = self.t + self.time_step
        return data


class NoiseModel(object):
    default_noise_scale = [N1AMP, N2AMP, N3AMP, N4AMP]

    def __init__(self, drop_probability=0.0, noise_array=None):
        self.noise = noise_array or np.asarray(
            NoiseModel.default_noise_scale, dtype=DTYPE
        )
        assert len(self.noise) == 4

    def gen_noise(self, data):
        assert len(data) == 4
        noise_values = []
        for i, d in enumerate(data):
            noise_values.append(
                self.noise[i] * data[i] * np.random.randn()
            )
        return np.asarray(noise_values, dtype=DTYPE)
