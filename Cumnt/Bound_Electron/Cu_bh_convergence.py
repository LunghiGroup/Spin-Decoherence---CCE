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

timespace_absolute = np.linspace(0, 30e-3, 2001)

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
Mx_baths = []
Mx_dipoles = []

default_bath_parameters = {
    'filepath': 'Path to Cu.xyz',
    'bath_type': 'hydrogen', # choose between 'electronic', 'hydrogen', 'deuterium'
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
        'n_clusters': {3:20000}
    }
    simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
    calc = simulator.setup_simulator()
    if MPI.COMM_WORLD.Get_rank()==0:
        print(calc)

    run = RunCalc(calc, **default_calc_parameters)
    result = run.run_calculation()

    Mx_orders.append(result)

for r in range(10,26,5):
    default_simulator_parameters = {
        'order': 2,
        'r_bath': r,
        'r_dipole': 8,
        'magnetic_field': [0, 0, 3300], # Magnetic field in Gauss
        'pulses': 1,
        'n_clusters': None
    }
    simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
    calc = simulator.setup_simulator()
    if MPI.COMM_WORLD.Get_rank()==0:
        print(calc)

    run = RunCalc(calc, **default_calc_parameters)
    result = run.run_calculation()

    Mx_baths.append(result)

for r in range(6,13,2):
    default_simulator_parameters = {
        'order': 2,
        'r_bath': 20,
        'r_dipole': r,
        'magnetic_field': [0, 0, 3300], # Magnetic field in Gauss
        'pulses': 1,
        'n_clusters': None
    }
    simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
    calc = simulator.setup_simulator()
    if MPI.COMM_WORLD.Get_rank()==0:
        print(calc)

    run = RunCalc(calc, **default_calc_parameters)
    result = run.run_calculation()

    Mx_dipoles.append(result)

np.savetxt('both_hydrogen_convergence_order_ET.dat', np.column_stack((timespace_absolute, Mx_orders[0], Mx_orders[1])))
np.savetxt('both_hydrogen_convergence_bath_ET.dat', np.column_stack((timespace_absolute, Mx_baths[0], Mx_baths[1], Mx_baths[2], Mx_baths[3])))
np.savetxt('both_hydrogen_convergence_dipole_ET.dat', np.column_stack((timespace_absolute, Mx_dipoles[0], Mx_dipoles[1], Mx_dipoles[2], Mx_dipoles[3])))
