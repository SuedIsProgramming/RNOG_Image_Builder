#!/data/condor_shared/users/ssued/RNOG_Image_Builder/venv/bin/python3

from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import os

root_dir = os.getcwd() # Should be main directory: RNOG_Image_Builder

import argparse # Argument parser required to add a simulation number suffix to the file name

parser = argparse.ArgumentParser(description='Argument for file differentiation')
parser.add_argument('sim_num', type=str,
                    help='Number of simulation')

args = parser.parse_args()

sim_num = args.sim_num

# Setup logging
from NuRadioReco.utilities.logging import _setup_logger
logger = _setup_logger(name="")

# define simulation volume (artificially close by to make them trigger)
volume = {
'fiducial_zmin':-3 * units.km,  # the ice sheet at South Pole is 2.7km deep
'fiducial_zmax': 0 * units.km,
'fiducial_rmin': 0 * units.km,
'fiducial_rmax': 1 * units.km}

# generate one event list at 1e19 eV with 1000 neutrinos
generate_eventlist_cylinder(f'{root_dir}/jobs/jobs_simulation_data/1e19_n1e3_{sim_num}.hdf5', 1e3, 1e19 * units.eV, 1e19 * units.eV, volume)