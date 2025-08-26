import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import sys

# Add parent directory to sys.path
sys.path.append('Path to Cu_class_T2.py')
from Cu_class_T2 import BathSetup, CenterSetup, SimulatorSetup, RunCalc
import pycce as pc
import ase
from mpi4py import MPI

np.set_printoptions(suppress=True, precision=5)

timespace_absolute = np.linspace(0, 0.5, 2001)

default_center_parameters = {
    'atens_file_path': 'Path to Cu.Atens',
    'gtens_file_path': 'Path to Cu.gtens',
    'spin_type': 'both', # choose between 'electron', 'nuclear', 'both'
    'alpha': 0,
    'beta': 4,
}

default_calc_parameters = {
    'timespace': timespace_absolute,
    'method': 'gcce',
    'nbstates': 0,
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

Mx_orders = []

default_bath_parameters = {
    'filepath': 'Path to Cu.xyz',
    'bath_type': 'deuterium', # choose between 'electronic', 'hydrogen', 'deuterium'
    'concentration': 1,
    'cell_size': 100,
    'seed': 8800
}

bath = BathSetup(**default_bath_parameters)
bath.create_bath()

center_pos = bath.sic.to_cartesian(bath.qpos)
center = CenterSetup(qpos=center_pos, **default_center_parameters)
cen = center.create_center()

for order in range(2,4):
    default_simulator_parameters = {
        'order': order,
        'r_bath': 20,
        'r_dipole': 8,
        'magnetic_field': [0, 0, 3300], # Magnetic field in Gauss
        'pulses': 1,
        'n_clusters': {2:22000, 3:5000}
    }
    simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
    calc = simulator.setup_simulator()
    if MPI.COMM_WORLD.Get_rank()==0:
        print(calc)

    run = RunCalc(calc, **default_calc_parameters)
    result = run.run_calculation()

    Mx_orders.append(result)

np.savetxt('both_deuterium_convergence_order_ET.dat', np.column_stack((timespace_absolute, Mx_orders[0], Mx_orders[1])))
