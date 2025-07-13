from __future__ import absolute_import, division, print_function
#import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
from utils.my_utils import get_rel_dir

# Setup logging
from NuRadioReco.utilities.logging import _setup_logger
logger = _setup_logger(name="")

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

rel_dir = get_rel_dir() 

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
    sim = mySimulation(inputfilename=f"{rel_dir}/1e19_n1e3.hdf5",
                                outputfilename=f"{rel_dir}/output.hdf5",
                                detectorfile=f"{rel_dir}/multistation.json",
                                outputfilenameNuRadioReco=f"{rel_dir}/output.nur",
                                config_file=f"{rel_dir}/config.yaml",
                                file_overwrite=True)
    sim.run()
