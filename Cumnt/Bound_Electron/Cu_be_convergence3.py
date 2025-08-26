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

timespace_absolute = np.linspace(0, 1e-4, 2001)

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
    'nbstates': 112,
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

Mx_orders = []
Mx_baths = []
Mx_dipoles = []
nsamples = 25
np.random.seed(8800)
seeds = list(np.random.randint(low=1,high=99999,size=nsamples))
seeds = seeds[10:15]
for seed in seeds:
    default_bath_parameters = {
    'filepath': 'Path to Cu.xyz',
    'bath_type': 'electronic', # choose between 'electronic', 'hydrogen', 'deuterium'
    'concentration': 0.3,
    'cell_size': 200,
    'seed': seed
    }

    bath = BathSetup(**default_bath_parameters)
    bath.create_bath()

    center_pos = bath.sic.to_cartesian(bath.qpos)
    center = CenterSetup(qpos=center_pos, **default_center_parameters)
    cen = center.create_center()

    Mx_order = []
    for order in range(2,5):
        default_simulator_parameters = {
        'order': order,
        'r_bath': 45,
        'r_dipole': 20,
        'magnetic_field': [0, 0, 3300], # Magnetic field in Gauss
        'pulses': 1,
        'n_clusters': {3: 2000, 4: 1000}
        }

        simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
        simulator.interlaced = True
        calc = simulator.setup_simulator()
        if MPI.COMM_WORLD.Get_rank()==0:
            print(calc)

        run = RunCalc(calc, **default_calc_parameters)
        result = run.run_calculation()

        Mx_order.append(result)
    Mx_orders.append(Mx_order)

    Mx_bath = []
    for r in range(30,61,10):
        default_simulator_parameters = {
        'order': 3,
        'r_bath': r,
        'r_dipole': 20,
        'magnetic_field': [0, 0, 3300], # Magnetic field in Gauss
        'pulses': 1,
        'n_clusters': {3: 2000}
        }

        simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
        simulator.interlaced = True
        calc = simulator.setup_simulator()
        if MPI.COMM_WORLD.Get_rank()==0:
            print(calc)

        run = RunCalc(calc, **default_calc_parameters)
        result = run.run_calculation()

        Mx_bath.append(result)
    Mx_baths.append(Mx_bath)

    Mx_dipole = []
    for r in range(15,26,5):
        default_simulator_parameters = {
        'order': 3,
        'r_bath': 45,
        'r_dipole': r,
        'magnetic_field': [0, 0, 3300], # Magnetic field in Gauss
        'pulses': 1,
        'n_clusters': {3: 2000}
        }

        simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
        simulator.interlaced = True
        calc = simulator.setup_simulator()
        if MPI.COMM_WORLD.Get_rank()==0:
            print(calc)

        run = RunCalc(calc, **default_calc_parameters)
        result = run.run_calculation()

        Mx_dipole.append(result)
    Mx_dipoles.append(Mx_dipole)

Mx_order_av_be = []
for i in range(3):
    Mx_sums = []
    for j in range(len(timespace_absolute)):
        Mx_sum = 0
        for k in range(len(seeds)):
            Mx_sum += Mx_orders[k][i][j]
        Mx_sums.append((1/nsamples)*Mx_sum)
    Mx_order_av_be.append(Mx_sums)

Mx_bath_av_be = []
for i in range(4):
    Mx_sums = []
    for j in range(len(timespace_absolute)):
        Mx_sum = 0
        for k in range(len(seeds)):
            Mx_sum += Mx_baths[k][i][j]
        Mx_sums.append((1/nsamples)*Mx_sum)
    Mx_bath_av_be.append(Mx_sums)

Mx_dipole_av_be = []
for i in range(3):
    Mx_sums = []
    for j in range(len(timespace_absolute)):
        Mx_sum = 0
        for k in range(len(seeds)):
            Mx_sum += Mx_dipoles[k][i][j]
        Mx_sums.append((1/nsamples)*Mx_sum)
    Mx_dipole_av_be.append(Mx_sums)

np.savetxt('both_electron_convergence_order_ET_3.dat', np.column_stack((timespace_absolute, Mx_order_av_be[0], Mx_order_av_be[1], Mx_order_av_be[2])))
np.savetxt('both_electron_convergence_bath_ET_3.dat', np.column_stack((timespace_absolute, Mx_bath_av_be[0], Mx_bath_av_be[1], Mx_bath_av_be[2], Mx_bath_av_be[3])))
np.savetxt('both_electron_convergence_dipole_ET_3.dat', np.column_stack((timespace_absolute, Mx_dipole_av_be[0], Mx_dipole_av_be[1], Mx_dipole_av_be[2])))
