from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
from utils.my_utils import get_rel_dir

rel_dir = get_rel_dir() 

# Setup logging
from NuRadioReco.utilities.logging import _setup_logger
logger = _setup_logger(name="")

# define simulation volume (artificially close by to make them trigger)
volume = {
'fiducial_zmin':-3 * units.km,  # the ice sheet at South Pole is 2.7km deep
'fiducial_zmax': 0 * units.km,
'fiducial_rmin': 0 * units.km,
'fiducial_rmax': 1 * units.km}

# generate one event list at 1e19 eV with 100 neutrinos
generate_eventlist_cylinder(f'{rel_dir}/1e19_n1e3.hdf5', 1e2, 1e19 * units.eV, 1e19 * units.eV, volume)