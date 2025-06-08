import NuRadioReco.modules.io.eventReader
from matplotlib import pyplot as plt
import numpy as np

def bin_hilbert(nbins,bin_method):
    # Stub that will eventually return a binned hilbert envelope
    return 0

class EventImage:
    def __init__(self,nur_file,nbins):
        self.nur_file = nur_file
        self.nbins = nbins

        event_reader = NuRadioReco.modules.io.eventReader.eventReader()
        event_reader.begin(nur_file)
        det = event_reader.get_detector() # Obtain detector description from .nur file
        if det is not None:
            nchannels = len(det._channels)
            nstations = len(det._stations)
        else:
            raise ValueError("Detector object is None. Cannot access '_channels' or '_stations'.")
        self.nrows = nchannels * nstations # Number of rows will be the total number of channels for all stations
        self.ncols = nbins # Number of columns is the number of bins

        for iE, event in enumerate(event_reader.run()):
            event_arr = np.zeros((self.nrows,self.ncols)) # Initialize an empty matrix
            for iStation, station in enumerate(event.get_stations()):
                for ch in station.iter_channels():
                    volts = ch.get_trace()
                    times = ch.get_times()

# For debugging
eimg = EventImage('/data/condor_shared/users/ssued/RNOG_Image_Builder/simulation/output.nur',24)