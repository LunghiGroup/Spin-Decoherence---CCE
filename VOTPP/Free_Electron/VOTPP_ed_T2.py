import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import sys

# Add parent directory to sys.path
sys.path.append('Path to VOTPP_class_T2.py')
from VOTPP_class_T2 import BathSetup, CenterSetup, SimulatorSetup, RunCalc
import pycce as pc
import ase
from mpi4py import MPI

np.set_printoptions(suppress=True, precision=5)

timespace = np.linspace(0, 0.5, 2001)


default_center_parameters = {
    'atens_file_path': 'Path to VOTPP_opt.Atens',
    'gtens_file_path': 'Path to VOTPP_opt.gtens',
    'spin_type': 'electron', # choose between 'electron', 'nuclear', 'both'
    'alpha': 0,
    'beta': 1,
}

default_calc_parameters = {
    'timespace': timespace,
    'method': 'gcce',
    'nbstates': 0, 
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

default_bath_parameters = {
    'filepath': 'Path to rotated_good.xyz',
    'bath_type': 'deuterium', # choose between 'electronic', 'hydrogen', 'nitrogen', 'carbon', 'deuterium'
    'concentration': 1, 
    'cell_size': 100, 
    'seed': 8800
}

bath = BathSetup(**default_bath_parameters)
bath.create_bath()

center_pos = bath.sic.to_cartesian(bath.qpos)
center = CenterSetup(qpos=center_pos, **default_center_parameters)
cen = center.create_center()

default_simulator_parameters = {
    'order': 2,
    'r_bath': 20,
    'r_dipole': 6,
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

np.savetxt('electron_deuterium_decoherence.dat', np.column_stack((timespace,result)))
