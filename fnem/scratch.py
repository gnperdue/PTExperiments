import torch
import numpy as np
import matplotlib.pyplot as plt
from sim.oscillators import Oscillator

osc = Oscillator(add_noise=True, drops=True)
plt.scatter(osc.ts(), osc.d1s(), c='r')
plt.scatter(osc.ts(), osc.d2s(), c='g')
plt.scatter(osc.ts(), osc.d3s(), c='b')
plt.scatter(osc.ts(), osc.d4s(), c='y')
plt.scatter(osc.ts(), osc.dts(), c='k')
# plt.scatter(osc.ts(), osc.dTs(), c='k')
plt.show()
for _ in range(10):
    osc.step()
    plt.scatter(osc.ts(), osc.dts(), c='r')
    plt.scatter(osc.ts(), osc.dTs(), c='b')
    plt.show()
