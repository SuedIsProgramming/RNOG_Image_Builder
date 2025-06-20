import NuRadioReco.modules.io.eventReader
from matplotlib import pyplot as plt
import numpy as np
import json

def bin(voltage,time,nbins,bin_method='mean'):
    total_time = time[-1] - time[0]
    bin_width = total_time * nbins
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

class EventImage:
    def __init__(self,nur_file,nbins=24,tot_time=6000):
        self.nur_file = nur_file
        self.nbins = nbins
        self.tot_time = tot_time 

        try: # Tries reading .nur file to obtain detector.
            event_reader = NuRadioReco.modules.io.eventReader.eventReader()
            event_reader.begin(nur_file, read_detector=True)
            det = event_reader.get_detector()  # Obtain detector description from .nur file
            if det is not None:
                self.nchannels = len(det._channels)
                self.nstations = len(det._stations)
                self.nrows = self.nchannels * self.nstations # Number of rows will be the total number of channels for all stations
            else:
                raise ValueError("Detector object is None. Cannot access '_channels' or '_stations'.")
        except Exception as e: # If an issue occurs, it will then simply try to read json.
            print('Issue encountered reading the .nur file, will read json instead')
            with open('simulation/multistation.json', 'r') as f:
                det = json.load(f)
                self.nchannels = len(det['channels'])
                self.nstations = len(det['stations'])
                self.nrows = self.nchannels * self.nstations

        self.ncols = nbins
        self.bin_time = tot_time / nbins
        for iE, event in enumerate(event_reader.run()):
            event_arr = np.zeros((self.nrows,self.ncols)) # Initialize an empty matrix
            for iStation, station in enumerate(event.get_stations()):
                for iChannel, channel in enumerate(station.iter_channels()):
                    volts_hilb = channel.get_hilbert_envelope()
                    times = channel.get_times()


# For debugging
eimg = EventImage('/data/condor_shared/users/ssued/RNOG_Image_Builder/simulation/output.nur',24)