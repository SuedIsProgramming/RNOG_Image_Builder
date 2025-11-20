#!/data/condor_shared/users/ssued/RNOG_Image_Builder/venv/bin/python3

"""
This script generates and saves event images for simulated particles detected by a multi-station detector array to a hdf5 file. The image consists of a 2d matrix with channels on the vertical axis
and binned time on the horizontal axis. Each image will be saved alongside a target label which will be used for machine learning purposes.
Workflow:
- Loads unique events from a .nur file and detector configuration from a JSON file.
- Extracts station and channel information from the configuration.
- For each event:
    - Aligns time arrays for all channels to the earliest time in the event.
    - Bins Hilbert envelope voltage data for each channel into time bins.
    - Constructs a 2D image array representing voltage vs. time for all channels and stations.
    - Obtains label value.
    - Saves image and label value to its corresponding event group in the .hdf5 file.
    
Key Parameters:
- TOT_TIME: Total time window (ns) for all traces for binning.
- N_BINS: Number of time bins.
- N_CHANNELS: Number of channels per station (from JSON).
- N_STATIONS: Number of stations (from JSON).
- BIN_MODE: Determines what value will be stored in each bin. MEAN: Save the mean hilbert value for each bin. MAX: Save the max hilbert value for each bin.
Dependencies:
- numpy
- matplotlib
- json
- utils.my_utils (get_unique_events, get_rel_dir)
Output:
- PNG images for each event, named as 'image_particle_{event_id}_detected_at_{stations_num}_stations.png' in the relevant directory.
Note:
- Assumes all stations have the same number of channels.
- Uses Hilbert envelope voltage data for visualization.
"""
import sys # Cannot escape this import :C

sys.path.append('/data/condor_shared/users/ssued/RNOG_Image_Builder')

from utils import my_utils
from NuRadioReco.framework import parameters
from matplotlib import pyplot as plt
import time
import numpy as np
import h5py
import json
import os

#print('Beginning step 3')

root_dir = os.getcwd() # Should be main directory: RNOG_Image_Builder

import argparse # Argument parser required to add a simulation number suffix to the file name

parser = argparse.ArgumentParser(description='Argument for file differentiation')
parser.add_argument('sim_num', type=str,
                    help='Number of simulation')

args = parser.parse_args()

sim_num = args.sim_num

events_unique = my_utils.get_unique_events(f'{root_dir}/jobs/jobs_simulation_data/output_{sim_num}.nur') # Obtain only the unique events for each particle

TOT_TIME = 7000
N_BINS = 1024 # Try to keep above this number
BIN_MODE = 'MEAN' # Can choose from MEAN, MAX

json_file = f'{root_dir}/jobs/jobs_simulation_data/multistation.json' # json file with detector info
albums_path = '/data/i3store/users/ssued/albums'

# max_index = my_utils.find_max_file_index(albums_path) Not required for jobs
hdf5_path = f'{albums_path}/mini-album{sim_num}.hdf5' # Save a new album

with open(json_file, 'r') as f:
    det = json.load(f) # Load detector
    N_CHANNELS = len(det['channels']) # Read number of channels (Assumes all stations have the same number of channels)
    N_STATIONS = len(det['stations']) # Assumes stations are all the same.
    STATION_IDS = []
    for st in range(1,N_STATIONS+1):
        STATION_IDS.append(det['stations'][f'{st}']['station_id'])

for iEvent, event in enumerate(events_unique):
    stations = list(event.get_stations())
    stations_num = len(stations)

    TOT_CHANNELS = N_CHANNELS * N_STATIONS # total number of channels, defines length of vertical axis

    # All traces are arranged in relation to a minimum time corresponding to the earliest detected signal accross all stations
    min_time = min( 
        channel.get_times()[0]
        for station in stations
        for channel in station.iter_channels()
    )

    image = np.zeros((TOT_CHANNELS, N_BINS)) # Initialize empty image
    
    time_edges = np.linspace(0, TOT_TIME, N_BINS + 1) # Setup edges of time bins

    # Main loop that assigns the trace from the array to bins on the matrix
    for iStation, station in enumerate(stations):
        iStation_id = STATION_IDS.index(station.get_id()) # Important, this will match each station with its corresponding index, regardless of the number of stations in the event *

        for iChannel, channel in enumerate(station.iter_channels()):
            curr_tot_channel = N_CHANNELS * iStation_id + iChannel # channel to store trace in
            # Obtain trace data
            time_arr = np.array(channel.get_times())
            hilb_arr = np.array(channel.get_hilbert_envelope())
            time_arr = time_arr - min_time

            time_bin_indices = np.digitize(time_arr,time_edges) # Returns the bins where each time_arr index belongs to.
            binned_hilb = [hilb_arr[time_bin_indices == i] for i in range(1, len(time_edges))] # Returns hilbert arrays for each bin

            # Process each array for each bin depending on mode
            for iBin, hilb_bin in enumerate(binned_hilb):
                if len(hilb_bin) == 0: # If no voltages for this bin, do nothing
                     continue
                elif BIN_MODE == 'MEAN': # If MEAN BIN MODE is selected, calculate the mean value
                    image[curr_tot_channel][iBin] = np.mean(hilb_bin)
                elif BIN_MODE == 'MAX': # If MEAN BIN MODE is selected, calculate the max value
                    image[curr_tot_channel][iBin] = np.max(hilb_bin)
                else:
                    raise ValueError(f"Unsupported BIN_MODE '{BIN_MODE}'. Only 'MEAN' and 'MAX' are supported.")

    # For now will use vertex as labels for training.
    vertex = next(event.get_particles()).get_parameter(parameters.particleParameters.vertex)

    # Save label + image to hdf5
    with h5py.File(hdf5_path, 'a') as file:
        #print(f'Saving event {iEvent + 1} to {hdf5_path}...')
        event = file.create_group(f'event{iEvent + 1}')
        event.create_dataset("image",data=image)
        event.create_dataset("label",data=vertex)

# * Events only include detected stations in get_stations(), thus, when looping through them it is important to keep this in mind. Looping through them will return the detected stations
# without care for their ID.