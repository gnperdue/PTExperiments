import torch
import numpy as np
import matplotlib.pyplot as plt
from sim.oscillators import Oscillator

from sim.engines import SimulationMachine
from sim.data_model import DataGenerator
from sim.data_model import NoiseModel
from sim.recorders import MachineStateTextRecorder
from policies.rule_based import SimpleRuleBased

np.random.seed(0)

dgen = DataGenerator()
nosgen = NoiseModel()
recorder = MachineStateTextRecorder('./test')
machine = SimulationMachine(setting=10.0, data_generator=dgen,
                            noise_model=nosgen, logger=recorder)
policy = SimpleRuleBased(time=0.0, setting=10.0, amplitude=10.0,
                         period=2.0, commands_array=machine.get_commands())

m1 = []
m2 = []
m3 = []
m4 = []
totals = []
heat = []
settings = []
ts = []
for i in range(2000):
    machine.step()
    t = machine.get_time()
    sensor_vals = machine.get_sensor_values()
    ts.append(t)
    for i, m in enumerate([m1, m2, m3, m4]):
        m.append(sensor_vals[i])
    totals.append(sum(sensor_vals))
    settings.append(machine.get_setting())
    heat.append(machine.get_heat())
    state = sensor_vals + [t]
    policy.set_state(state)
    command = policy.compute_action()
    machine.update_machine(command)
    policy.update_setting(command)

fig = plt.Figure(figsize=(10, 6))
gs = plt.GridSpec(1, 4)
ax1 = plt.subplot(gs[0])
ax1.scatter(ts, m1, c='r')
ax1.scatter(ts, m2, c='g')
ax1.scatter(ts, m3, c='b')
ax1.scatter(ts, m4, c='y')
ax2 = plt.subplot(gs[1])
ax2.scatter(ts, totals, c='k')
ax3 = plt.subplot(gs[2])
ax3.scatter(ts, heat, c='k')
ax4 = plt.subplot(gs[3])
ax4.scatter(ts, settings, c='k')
plt.show()
