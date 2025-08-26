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

default_center_parameters = {
    'atens_file_path': 'Path to VOTPP_opt.Atens',
    'gtens_file_path': 'Path to VOTPP_opt.gtens',
    'spin_type': 'electron', # choose between 'electron', 'nuclear', 'both'
    'alpha': 0,
    'beta': 1,
}

Mx_ee = []
np.random.seed(8800)
seeds = list(np.random.randint(low=1,high=99999,size=50))

timespace = np.linspace(0, 4e-5, 2001)
default_calc_parameters = {
    'timespace': timespace,
    'method': 'gcce',
    'nbstates': 112,
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

for i in range(len(seeds)):
    default_bath_parameters = {
        'filepath': 'Path to rotated_good.xyz',
        'bath_type': 'electronic', # choose between 'electronic', 'hydrogen', 'nitrogen', 'carbon', 'deuterium'
        'concentration': 0.55,
        'cell_size': 200,
        'seed': seeds[i]
    }

    bath = BathSetup(**default_bath_parameters)
    bath.create_bath()

    center_pos = bath.sic.to_cartesian(bath.qpos)
    center = CenterSetup(qpos=center_pos, **default_center_parameters)
    cen = center.create_center()

    default_simulator_parameters = {
        'order': 3,
        'r_bath': 35,
        'r_dipole': 15,
        'magnetic_field': [0, 0, 3300], # Magnetic field in Gauss
        'pulses': 1,
        'n_clusters': None
    }
    simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
    simulator.interlaced = True
    calc = simulator.setup_simulator()
    if MPI.COMM_WORLD.Get_rank()==0:
        print(calc)

    run = RunCalc(calc, **default_calc_parameters)
    result = run.run_calculation()

    Mx_ee.append(result)

Mx_av_ee = []
for j in range(len(timespace)):
    Mx_sum = 0
    for k in range(len(seeds)):
        Mx_sum += Mx_ee[k][j]
    Mx_av_ee.append(Mx_sum/len(seeds))

np.savetxt('electron_electron_concentrations55.dat', np.column_stack((timespace, Mx_av_ee)))
