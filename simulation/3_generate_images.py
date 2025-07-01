from matplotlib import pyplot as plt
from utils.my_utils import get_unique_events, get_rel_dir
import numpy as np
import json

rel_dir = get_rel_dir()
events_unique = get_unique_events(f'{rel_dir}/output.nur')

TOT_TIME = 6000
N_BINS = 1024 # Try to keep above this number

json_file = f'{rel_dir}/multistation.json'

with open(json_file, 'r') as f:
            print(f'Issue encountered reading the .nur file, will read from {json_file} instead')
            with open(json_file, 'r') as f:
                det = json.load(f)
                N_CHANNELS = len(det['channels'])
                N_STATIONS = len(det['stations']) # Assumes stations are all the same.
                STATION_IDS = []
                CHANNEL_IDS = []

                for st in range(1,N_STATIONS+1):
                    STATION_IDS.append(det['stations'][f'{st}']['station_id'])
                    for key in det['channels'].keys():
                        CHANNEL_IDS.append(key)

for event in events_unique:
    stations = list(event.get_stations()) #MIGHT NOT NEED THIS
    stations_num = len(stations)
    # # Get the number of channels for each station (assuming all stations have the same number of channels)
    # channels_per_station = [len(list(station.iter_channels())) for station in stations]
    TOT_CHANNELS = N_CHANNELS * N_STATIONS # sum(channels_per_station)

    min_time = min(
        channel.get_times()[0]
        for station in stations
        for channel in station.iter_channels()
    )

    image = np.zeros((TOT_CHANNELS, N_BINS))
    
    time_edges = np.linspace(0, TOT_TIME, N_BINS + 1)

    for iStation, station in enumerate(stations):
        for iChannel, channel in enumerate(station.iter_channels()):
            curr_tot_channel = N_CHANNELS * iStation + iChannel
            time_arr = np.array(channel.get_times())
            hilb_arr = np.array(channel.get_hilbert_envelope())
            time_arr = time_arr - min_time
            for iTime, time in enumerate(time_arr):
                for iTime_edge, time_edge in enumerate(time_edges):
                    if time < time_edge:
                        image[curr_tot_channel][iTime_edge] = hilb_arr[iTime]
                        break
    
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
    fig.savefig(f'{rel_dir}/image_particle_{event.get_id()}_detected_at_{stations_num}_stations.png', bbox_inches='tight')
    plt.close(fig)
