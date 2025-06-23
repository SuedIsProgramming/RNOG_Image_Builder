import NuRadioReco.modules.io.eventReader
from matplotlib import pyplot as plt
import numpy as np
import json

class EventImage:
    def __init__(self,nur_file,nbins=24,tot_time=6000):
        self.nur_file = nur_file
        self.nbins = nbins
        self.tot_time = tot_time
        self.event_reader = NuRadioReco.modules.io.eventReader.eventReader()

        self.nchannels, self.nstations, self.adc_n_samples = self._get_station_properties(self.nur_file, 'simulation/multistation.json')
        self.nrows = self.nchannels * self.nstations # Number of rows will be the total number of channels for all stations
        self.ncols = nbins
        self.bin_time = tot_time / nbins
        self.matrix = self._create_matrix()

    def _get_station_properties(self, nur_file, json_file):
        try: # Tries reading .nur file to obtain detector.
            self.event_reader.begin(nur_file, read_detector=True)
            det = self.event_reader.get_detector()  # Obtain detector description from .nur file
            if det is not None:
                nchannels = len(det._channels)
                nstations = len(det._stations)
                adc_n_samples = None #Should figure out if they ever fix this.
            else:
                raise ValueError("Detector object is None. Cannot access '_channels' or '_stations'.")
        except Exception as e: # If an issue occurs, it will then simply try to read json.
            print(f'Issue encountered reading the .nur file, will read from {json_file} instead')
            with open(json_file, 'r') as f:
                det = json.load(f)
                nchannels = len(det['channels'])
                nstations = len(det['stations'])
                adc_n_samples = det['channels']['1']['adc_n_samples'] # Right now using the same adc_n_samples for each channel.
        return nchannels, nstations, adc_n_samples

    def _create_matrix(self):
        for iE, event in enumerate(self.event_reader.run()):
            event_matrix = np.zeros((self.nrows,self.ncols)) # Initialize an empty matrix
            start_time = float('inf')
            for iStation, station in enumerate(event.get_stations()):
                for iChannel, channel in enumerate(station.iter_channels()):

                    times = channel.get_times()
                    if times[0] < start_time:
                        start_time = times[0]

    def _bin(self, voltage,time,nbins,bin_method='mean'):
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



# For debugging
eimg_test = EventImage('/data/condor_shared/users/ssued/RNOG_Image_Builder/simulation/output.nur',24)
print(eimg_test.adc_n_samples)