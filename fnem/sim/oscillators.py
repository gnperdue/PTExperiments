import numpy as np


class Oscillator(object):
    amp = [10.0, 1.0, 0.5, 0.1]
    frq = [np.pi * 1.0, np.pi * 2.0, np.pi * 3.0, np.pi * 4.0]
    nos = [0.4, 0.1, 0.05, 0.01]
    drop_prob = 0.05

    def __init__(self, span=0.5, spacing=0.01,
                 add_noise=False, drops=False):
        self.t = 0
        self.nsteps = int(span / spacing)
        self.spacing = spacing
        self.add_noise = add_noise
        self.drops = drops
        self.times, self.d1, self.d2, self.d3, self.d4, self.dT = \
            [], [], [], [], [], []
        for i in range(self.nsteps):
            self._add_data_point()
            self.t = self.t + self.spacing

    def _add_data_point(self):
        if len(self.times) >= self.nsteps:
            for a in [self.times, self.d1, self.d2, self.d3, self.d4, self.dT]:
                a.pop(0)
        self.times.append(self.t)
        raw_vs = []
        nos_vs = []
        for i, a in enumerate([self.d1, self.d2, self.d3, self.d4]):
            raw_vs.append(
                Oscillator.amp[i] * np.cos(Oscillator.frq[i] * self.t)
            )
            nos_vs.append(Oscillator.nos[i] * np.random.randn())
            if self.drops and np.random.random() < Oscillator.drop_prob:
                a.append(0.0)
            else:
                a.append(raw_vs[-1] + nos_vs[-1] if self.add_noise else 0.0)
        self.dT.append(sum(raw_vs))

    def step(self):
        self._add_data_point()
        self.t = self.t + self.spacing

    def ts(self):
        return np.asarray(self.times)

    def d1s(self):
        return np.asarray(self.d1)

    def d2s(self):
        return np.asarray(self.d2)

    def d3s(self):
        return np.asarray(self.d3)

    def d4s(self):
        return np.asarray(self.d4)

    def dts(self):
        return np.asarray(self.d1) + np.asarray(self.d2) \
            + np.asarray(self.d3) + np.asarray(self.d4)

    def dTs(self):
        return np.asarray(self.dT)
