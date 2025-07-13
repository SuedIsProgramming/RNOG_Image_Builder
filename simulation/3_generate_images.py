"""
This script generates and saves event images for simulated particles detected by a multi-station detector array. The image consists of a 2d matrix with channels on the vertical axis
and binned time on the horizontal axis.
Workflow:
- Loads unique events from a .nur file and detector configuration from a JSON file.
- Extracts station and channel information from the configuration.
- For each event:
    - Aligns time arrays for all channels to the earliest time in the event.
    - Bins Hilbert envelope voltage data for each channel into time bins.
    - Constructs a 2D image array representing voltage vs. time for all channels and stations.
    - Visualizes the image using matplotlib, with axes labeled for time, channel, and station.
    - Saves the resulting image as a PNG file.
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
from matplotlib import pyplot as plt
from utils.my_utils import get_unique_events, get_rel_dir
import numpy as np
import json

print('Beginning step 3')

rel_dir = get_rel_dir() # Obtain the relative directory path for relative file administration
events_unique = get_unique_events(f'{rel_dir}/output.nur') # Obtain only the unique events for each particle

TOT_TIME = 6000
N_BINS = 1024 # Try to keep above this number
BIN_MODE = 'MEAN' # Can choose from MEAN, MAX

json_file = f'{rel_dir}/multistation.json' # json file with detector info

with open(json_file, 'r') as f:
            print(f'Issue encountered reading the .nur file, will read from {json_file} instead')
            with open(json_file, 'r') as f:
                det = json.load(f) # Load detector
                N_CHANNELS = len(det['channels']) # Read number of channels (Assumes all stations have the same number of channels)
                N_STATIONS = len(det['stations']) # Assumes stations are all the same.
                STATION_IDS = []
                CHANNEL_IDS = []

                # Populate station and channel ids for future plotting and indexing
                for st in range(1,N_STATIONS+1):
                    STATION_IDS.append(det['stations'][f'{st}']['station_id'])
                    for key in det['channels'].keys():
                        CHANNEL_IDS.append(key)

for event in events_unique:
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
            
            # Old code, will delete once Im sure new code works well
            # Loop through time array (TODO: As is, will continuously save over itself until time is greater than time edge for each time bin)
            # Want to make it save the mean. I believe right now it saves the right most time value for each time edge
            # for iTime, time in enumerate(time_arr):
            #     for iTime_edge, time_edge in enumerate(time_edges):
            #         if time < time_edge: # If time is smaller than time edge
            #             image[curr_tot_channel][iTime_edge] = hilb_arr[iTime] # Save the hilbert voltage at that time and continue with next time.
            #             break

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

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(image, aspect='auto', interpolation='none')
    ax.set_xlabel(f'Time Bin (Each bin is {TOT_TIME / N_BINS:.2f} ns)')
    ax.set_ylabel('Channel')
    ax.set_yticks(np.arange(len(CHANNEL_IDS)))
    ax.set_yticklabels(CHANNEL_IDS)
    ax_twin_x = ax.twiny()
    ax_twin_x.set_xlim(ax.get_xlim())
    ax_twin_x.set_xlabel('Time (ns)')
    time_labels = np.linspace(0, TOT_TIME, num=5)
    ax_twin_x.set_xticks(np.linspace(0, N_BINS, num=5))
    ax_twin_x.set_xticklabels([f"{t:.1f}" for t in time_labels])
    ax_twin_y = ax.twinx()
    ax_twin_y.set_ylim(ax.get_ylim())
    ax_twin_y.set_ylabel('Station ID')
    ax_twin_y.set_yticks([i * N_CHANNELS + N_CHANNELS / 2 - 0.5 for i in range(N_STATIONS)])
    ax_twin_y.set_yticklabels(STATION_IDS)
    ax.set_title(f'Event Image for Simulated Particle {event.get_id()}')
    cbar = fig.colorbar(im, ax=ax, label='Voltage (V)')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    for iStations in range(N_STATIONS):
        ax.axhline(y=iStations*N_CHANNELS-0.5, color='w', linewidth=2, xmin=0, xmax=1)
        for iChannel in range(N_CHANNELS):
             ax.axhline(y=iStations*N_CHANNELS+iChannel-0.5, color='w', linewidth=1, xmin=0, xmax=1, alpha=0.5)
    ax.set_yticks(np.arange(image.shape[0]))

    # Saving
    fpath = f'images_and_traces/image_particle_{event.get_id()}_detected_at_{stations_num}_stations_mode_{BIN_MODE}.png'

    print(f' Saving {fpath} ...')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)

# * Events only include detected stations in get_stations(), thus, when looping through them it is important to keep this in mind. Looping through them will return the detected stations
# without care for their ID.