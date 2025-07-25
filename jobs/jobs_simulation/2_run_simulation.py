#!/data/condor_shared/users/ssued/RNOG_Image_Builder/venv/bin/python3

from __future__ import absolute_import, division, print_function
#import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
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

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)

    def _detector_simulation_trigger(self, evt, station, det):
        # first run a simple threshold trigger
        simpleThreshold.run(evt, station, det,
                             threshold=3 * self._Vrms,
                             triggered_channels=None,  # run trigger on all channels
                             number_concidences=1,
                             trigger_name='simple_threshold')  # the name of the trigger

        # run a high/low trigger on the 4 downward pointing LPDAs
        highLowThreshold.run(evt, station, det,
                                    threshold_high=4 * self._Vrms,
                                    threshold_low=-4 * self._Vrms,
                                    triggered_channels=None,  # select the LPDA channels | Santi: Will trigger for all channels.
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_4.1sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

# parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
# parser.add_argument('inputfilename', type=str,
#                     help='path to NuRadioMC input event list')
# parser.add_argument('detectordescription', type=str,
#                     help='path to file containing the detector description')
# parser.add_argument('config', type=str,
#                     help='NuRadioMC yaml config file')
# parser.add_argument('outputfilename', type=str,
#                     help='hdf5 output filename')
# parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
#                     help='outputfilename of NuRadioReco detector sim file')
# args = parser.parse_args()

if __name__ == "__main__":
    sim = mySimulation(inputfilename=f"{root_dir}/jobs/jobs_simulation_data/1e19_n1e3_{sim_num}.hdf5",
                                outputfilename=f"{root_dir}/jobs/jobs_simulation_data/output_{sim_num}.hdf5",
                                detectorfile=f"{root_dir}/jobs/jobs_simulation_data/multistation.json",
                                outputfilenameNuRadioReco=f"{root_dir}/jobs/jobs_simulation_data/output_{sim_num}.nur",
                                config_file=f"{root_dir}/jobs/jobs_simulation_data/config.yaml",
                                file_overwrite=True)
    sim.run()
