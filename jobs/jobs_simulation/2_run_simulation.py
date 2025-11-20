#!/data/condor_shared/users/ssued/RNOG_Image_Builder/venv/bin/python3

from __future__ import absolute_import, division, print_function
#import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import time
import psutil
import os

start_time = time.time()

root_dir = os.getcwd() # Should be main directory: RNOG_Image_Builder

import argparse # Argument parser required to add a simulation number suffix to the file name

parser = argparse.ArgumentParser(description='Argument for file differentiation')
parser.add_argument('sim_num', type=str,
                    help='Number of simulation')

args = parser.parse_args()

sim_num = args.sim_num

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

print('Before:')
print_memory_usage()

# Setup logging
from NuRadioReco.utilities.logging import _setup_logger
logger = _setup_logger(name="")

# initialize detector sim modules
hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()

class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        hardware_response.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        highLowThreshold.run(evt, station, det,
                                    threshold_high=20 * units.mV,
                                    threshold_low=-20 * units.mV,
                                    triggered_channels=[0, 1],
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='main_trigger'
                             )

if __name__ == "__main__":
    sim = mySimulation(inputfilename=f"{root_dir}/jobs/jobs_simulation_data/1e19_n1e3_{sim_num}.hdf5",
                                outputfilename=f"{root_dir}/jobs/jobs_simulation_data/output_{sim_num}.hdf5",
                                detectorfile=f"{root_dir}/jobs/jobs_simulation_data/RNO_four_stations.json",
                                outputfilenameNuRadioReco=f"{root_dir}/jobs/jobs_simulation_data/output_{sim_num}.nur",
                                config_file=f"{root_dir}/jobs/jobs_simulation_data/config.yaml",
                                file_overwrite=True)
    sim.run()

end_time = time.time()

print('After:')
print_memory_usage()
print(f'Runtime : {end_time - start_time} s')