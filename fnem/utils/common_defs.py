import numpy as np
import os
import pathlib

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

source_path = pathlib.Path(os.environ['HOME'])
source_path = source_path/'Dropbox/ArtificialIntelligence/InterestingPyTorch'
source_path = source_path/'PTExperiments/fnem/reference_files'
DEFAULT_SOURCE_PATH = source_path

MACHINE_WITH_RULE_LOG_TEMPLATE = 'log_machinewithrule_%d'
MACHINE_WITH_RULE_PLT_TEMPLATE = 'plt_machinewithrule_%d'
DATASET_MACHINE_LOG_TEMPLATE = 'log_dataset_machine_%d'

REFERENCE_TSTAMP1 = 1545976343
MACHINE_WITH_RULE_REFERNECE_LOG = \
    (MACHINE_WITH_RULE_LOG_TEMPLATE % REFERENCE_TSTAMP1) + '.csv.gz'
MACHINE_WITH_RULE_REFERNECE_LOG = \
    pathlib.Path('./reference_files')/MACHINE_WITH_RULE_REFERNECE_LOG
MACHINE_WITH_RULE_REFERENCE_PLT = \
    (MACHINE_WITH_RULE_PLT_TEMPLATE % REFERENCE_TSTAMP1) + '.pdf'
MACHINE_WITH_RULE_REFERENCE_PLT = \
    pathlib.Path('./reference_files')/MACHINE_WITH_RULE_REFERENCE_PLT
