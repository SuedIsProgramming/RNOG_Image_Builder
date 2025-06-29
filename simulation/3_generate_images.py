from matplotlib import pyplot as plt
from utils.my_utils import get_unique_events, get_rel_dir
import numpy as np
import json

def _bin(voltage,time,nbins,bin_method='mean'):
    total_time = time[-1] - time[0]
    bin_width = total_time / nbins
    bin_edges = np.arange(time[0],time[-1] + bin_width, bin_width)
    #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 Is there any use of having the centers?

    bin_indices = np.digitize(time, bin_edges)

    bin_voltage = np.zeros(len(bin_edges) - 1)

    for i in range(1, len(bin_edges)):
        voltages_in_bin = voltage[bin_indices == i]
        if voltages_in_bin.size > 0:
            if bin_method == 'max':
                bin_voltage[i - 1] = np.max(voltages_in_bin)
            elif bin_method == 'mean':
                bin_voltage[i - 1] = np.mean(voltages_in_bin)

    return bin_voltage, (time[0],time[-1])

rel_dir = get_rel_dir()
events_unique = get_unique_events(f'{rel_dir}/output.nur')

TOT_TIME = 6000
N_BINS = 1024

json_file = f'{rel_dir}/multistation.json'
with open(json_file, 'r') as f:
            print(f'Issue encountered reading the .nur file, will read from {json_file} instead')
            with open(json_file, 'r') as f:
                det = json.load(f)
                N_CHANNELS = len(det['channels'])
                N_STATIONS = len(det['stations']) # Assumes stations are all the same.

album = []

for event in events_unique:
    stations = list(event.get_stations()) #MIGHT NOT NEED THIS
    # stations_num = len(stations)
    # # Get the number of channels for each station (assuming all stations have the same number of channels)
    # channels_per_station = [len(list(station.iter_channels())) for station in stations]
    TOT_CHANNELS = N_CHANNELS * N_STATIONS # sum(channels_per_station)

    min_time = min(
        channel.get_times()[0]
        for station in stations
        for channel in station.iter_channels()
    )

    image = np.zeros((TOT_CHANNELS, N_BINS))
    
    for station in stations:
        for channel in station.iter_channels():
            time_arr = np.array(channel.get_times())
            time_arr = time_arr - min_time
            # print(time_arr)






    

