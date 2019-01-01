import numpy as np

# data amplitudes
D1AMP, D2AMP, D3AMP, D4AMP = 10.0, 1.0, 0.5, 0.1
# data update frequencies / pi
D1FRQ, D2FRQ, D3FRQ, D4FRQ = 1.0, 0.1, 3.0, 10.0
DTYPE = np.float32
# noise amplitudes
N1AMP, N2AMP, N3AMP, N4AMP = 0.05, 0.04, 0.02, 0.01

# machine update "dial settings" (if allowing only discreet updates)
DEFAULT_COMMANDS = np.array([-0.5, -0.375, -0.25, -0.125, 0.0,
                             0.125, 0.25, 0.375, 0.5], dtype=DTYPE)

# strategies for running a trainer class
RUN_MODES = [
    'RUN-TRAINED-HISTORICAL', 'RUN-TRAINED-LIVE', 'TRAIN-HISTORICAL',
    'TRAIN-LIVE'
]
