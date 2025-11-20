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
from pathlib import Path
import numpy as np
import h5py
import json
import time
import os
import shutil

start_time = time.time()

#print('Beginning step 3')

root_dir = os.getcwd() # Should be main directory: RNOG_Image_Builder

import argparse # Argument parser required to add a simulation number suffix to the file name

parser = argparse.ArgumentParser(description='Argument for file differentiation')
parser.add_argument('sim_num', type=str,
                    help='Number of simulation')

args = parser.parse_args()

sim_num = args.sim_num
nurfile_path = f'{root_dir}/jobs/jobs_simulation_data/output_{sim_num}.nur'

try:
    events_unique = my_utils.get_unique_events(nurfile_path) # Obtain only the unique events for each particle
except FileNotFoundError:
    print(f"File not found: {root_dir}/jobs/jobs_simulation_data/output_{sim_num}.nur")
    sys.exit(1)

TOT_TIME = 8192
N_BINS = 1024 # Try to keep above this number
BIN_MODE = 'MEAN' # Can choose from MEAN, MAX
V_CUTTING = True
SPHERICAL_OUTPUT = False
NORMALIZE_OUTPUT = False

json_file = f'{root_dir}/jobs/jobs_simulation_data/RNO_four_stations.json' # json file with detector info
album_name = my_utils.get_album_name() 
albums_path = f'/data/i3store/users/ssued/albums/{album_name}'
n1_album_path = f'{albums_path}/n1_album'
n2_album_path = f'{albums_path}/n2_album'
n3_album_path = f'{albums_path}/n3_album'
n4_album_path = f'{albums_path}/n4_album'
Path(albums_path).mkdir(parents=True, exist_ok=True)
Path(n1_album_path).mkdir(parents=True, exist_ok=True)
Path(n2_album_path).mkdir(parents=True, exist_ok=True)
Path(n3_album_path).mkdir(parents=True, exist_ok=True)
Path(n4_album_path).mkdir(parents=True, exist_ok=True)

# max_index = my_utils.find_max_file_index(albums_path) Not required for jobs

with open(json_file, 'r') as f:
    det = json.load(f) # Load detector
    N_CHANNELS = len(det['channels']) # Read number of channels (Assumes all stations have the same number of channels)
    N_STATIONS = len(det['stations']) # Assumes stations are all the same.
    STATION_IDS = []
    for st in range(1,N_STATIONS+1):
        STATION_IDS.append(det['stations'][f'{st}']['station_id'])

for iEvent, event in enumerate(events_unique):
    skip_event = False # For skipping events that do not meet criteria
    stations = list(event.get_stations())
    stations_num = len(stations)

    hdf5_path = f'{albums_path}/n{stations_num}_album/mini-album{sim_num}.hdf5' # Save an album wrt multiplicity

    station_ids = [station.get_id() for station in stations]
    print(f'Event detected at {stations_num} stations {station_ids}. Saving to corresponding album at {hdf5_path}')

    # All traces are arranged in relation to a minimum time corresponding to the earliest detected signal accross all stations
    min_time = min( 
        channel.get_times()[0]
        for station in stations
        for channel in station.iter_channels()
    )

    image = np.zeros((N_CHANNELS, N_BINS, N_STATIONS)) # Initialize empty image
    time_edges = np.linspace(0, TOT_TIME, N_BINS + 1) # Setup edges of time bins
    max_time_window = np.max(time_edges)
    max_v = -1
    max_v_evt = 0
    max_v_sta = 0
    max_v_ch = 0

    # Main loop that assigns the trace from the array to bins on the matrix
    for iStation, station in enumerate(stations):
        iStation_id = STATION_IDS.index(station.get_id()) # Important, this will match each station with its corresponding index, regardless of the number of stations in the event *

        for iChannel, channel in enumerate(station.iter_channels()):
            # Obtain trace data
            time_arr = np.array(channel.get_times())
            hilb_arr = np.array(channel.get_hilbert_envelope())
            print(f'Max hilbert voltage: {np.max(hilb_arr)} V')

            if np.max(hilb_arr) > max_v:
                max_v = np.max(hilb_arr)
                max_v_evt = iEvent
                max_v_sta = iStation_id
                max_v_ch = iChannel
            
            if V_CUTTING and np.max(hilb_arr) > 1:
                # For debugging, will tell me the largest voltage value. Additionally, will save extraordinarily strong pulses.
                print('Hilbert Voltage exceeds 1V! Saving...')
                new_nurfile_path = f'{albums_path}/interesting_nurfiles/interesting_output_{sim_num}_volt-{max_v}_evt-{max_v_evt}_st-{max_v_sta}_ch-{max_v_ch}.nur'
                if not os.path.exists(new_nurfile_path):
                    os.makedirs(new_nurfile_path, exist_ok=True)
                    shutil.copy(nurfile_path, new_nurfile_path)
                else:
                    print('nurfile already exists, skipping...')
                skip_event = True
                break # If voltage too high break out of channel loop

            time_arr = time_arr - min_time
            max_rel_time = time_arr[-1]

            if max_rel_time > max_time_window: # For debugging, will let me know how big I should make the time window.
                print(f'Error! Events detected outside of stipulated time window:')
                print(f'Time window: {max_time_window} ns')
                print(f'Relative Time Maximum: {max_rel_time} ns')
                skip_event = True
                break # If time window too high, break out of time window

            time_bin_indices = np.digitize(time_arr,time_edges) # Returns the bins where each time_arr index belongs to.
            binned_hilb = [hilb_arr[time_bin_indices == i] for i in range(1, len(time_edges))] # Returns hilbert arrays for each bin

            # Process each array for each bin depending on mode
            for iBin, hilb_bin in enumerate(binned_hilb):
                if len(hilb_bin) == 0: # If no voltages for this bin, do nothing
                     continue
                elif BIN_MODE == 'MEAN': # If MEAN BIN MODE is selected, calculate the mean value
                    image[iChannel][iBin][iStation_id] = np.mean(hilb_bin)
                elif BIN_MODE == 'MAX': # If MEAN BIN MODE is selected, calculate the max value
                    image[iChannel][iBin][iStation_id] = np.max(hilb_bin)
                else:
                    raise ValueError(f"Unsupported BIN_MODE '{BIN_MODE}'. Only 'MEAN' and 'MAX' are supported.")
        if skip_event: # Skip this station loop
            break
    
    if skip_event: # Continue with next event
        continue

    # For now will use vertex as labels for training.
    vertex = next(event.get_particles()).get_parameter(parameters.particleParameters.vertex)

    if SPHERICAL_OUTPUT:
        x,y,z = vertex

        r = np.sqrt(x**2+y**2+z**2)
        phi = np.arctan2(y, x) # Will output from [-pi,pi] 
        if r == 0:
            theta = 0
        else:
            theta = np.arccos(z/r)

        # Because z is always negative, theta will be between [pi/2, pi]
        if NORMALIZE_OUTPUT:
            r_mean = 4743 / 2 # Assuming r is uniform, which is not true, but close enough.
            r_std = (1/np.sqrt(12)) * (4743 / 2)
            r = (r - r_mean) / r_std

            # phi mean is 0
            phi_std = (1/np.sqrt(12)) * (2*np.pi)
            phi = phi/ phi_std

            theta_mean = 3*np.pi / 4
            theta_std = (1/np.sqrt(12)) * (np.pi / 2)
            theta = (theta - theta_mean) / theta_std

        vertex = [r,phi,theta]

    # Save label + image to hdf5
    with h5py.File(hdf5_path, 'a') as file:
        print(f'Saving event {iEvent + 1} to {hdf5_path}...')
        event = file.create_group(f'event{iEvent + 1}')
        event.create_dataset("image",data=image)
        event.create_dataset("label",data=vertex)

# * Events only include detected stations in get_stations(), thus, when looping through them it is important to keep this in mind. Looping through them will return the detected stations
# without care for their ID.

end_time = time.time()
print(f'Runtime : {end_time - start_time} s')
