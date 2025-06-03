from matplotlib import pyplot as plt

import NuRadioReco.modules.io.eventReader
event_reader = NuRadioReco.modules.io.eventReader.eventReader()

file = 'output.nur'
event_reader.begin(file)
for iE, event in enumerate(event_reader.run()):
    primary = event.get_primary()

    for iStation, station in enumerate(event.get_stations()):

        # a fig and axes for our waveforms
        fig, axs = plt.subplots(8, 1, figsize=(5,20))

        # this loops through "mock data" (with noise added, etc.)
        for ch in station.iter_channels():
            volts = ch.get_trace()
            times = ch.get_times()
            axs[ch.get_id()].plot(times, volts, label='V noise') # type: ignore
            axs[ch.get_id()].set_title(f"Station {station.get_id()}, Channel {ch.get_id()}") # type: ignore
        
        # this loops through *MC truth* waveforms (before noise was added)
        # this may prove useful at some point
        if station.has_sim_station():
            sim_station = station.get_sim_station()
            for sim_ch in sim_station.iter_channels():
                volts = sim_ch.get_trace()
                times = sim_ch.get_times()
                axs[sim_ch.get_id()].plot(times, volts, '--',label='V raw') # type: ignore
    
        for ax in axs:
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Voltage [V]")
            ax.legend()

        plt.tight_layout()

        fig.savefig(f"traces_{iE}.png") # save the traces
