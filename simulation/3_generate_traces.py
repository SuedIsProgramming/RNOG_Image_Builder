from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

import NuRadioReco.modules.io.eventReader
event_reader = NuRadioReco.modules.io.eventReader.eventReader()

file = 'output.nur'
event_reader.begin(file)
for iE, event in enumerate(event_reader.run()):
    primary = event.get_primary()
    iP = primary.get_id()

    for iStation, station in enumerate(event.get_stations()):
        stations = list(event.get_stations())
        stations_num = len(stations)
        channels = list(station.iter_channels())
        channels_num = len(channels)

        # a fig and axes for our waveforms
        fig, axs = plt.subplots(stations_num*channels_num, 1, figsize=(5,20)) # subplots will scale with num of total channels

        # this loops through "mock data" (with noise added, etc.)
        for ch in station.iter_channels():
            volts = ch.get_trace()
            times = ch.get_times()
            tot_time = (times[-1] - times[0]) + 1 # Have to add one ns because first element starts at 1 ns.
            tot_time_box = AnchoredText(f'$t_{{tot}}:$ {int(tot_time)} ns',loc=1)
            axs[ch.get_id()].add_artist(tot_time_box) # Add total time of trace (Only calculated for true V, but should be the same regardless)
            axs[ch.get_id()].plot(times, volts, label='V') # type: ignore
            axs[ch.get_id()].set_title(f"Particle number {iP} | Station {station.get_id()}, Channel {ch.get_id()}") # type: ignore
        
        # this loops through *MC truth* waveforms (before noise was added)
        # this may prove useful at some point
        if station.has_sim_station():
            sim_station = station.get_sim_station()
            for sim_ch in sim_station.iter_channels():
                volts = sim_ch.get_trace()
                times = sim_ch.get_times()
                if sim_ch.get_ray_tracing_solution_id() == 0: # If else check to add direct and reflected traces.
                    axs[sim_ch.get_id()].plot(times, volts, '--',label=f'Vsim direct',color='tab:green') # type: ignore
                else:
                    axs[sim_ch.get_id()].plot(times, volts, '--',label=f'Vsim reflected',color='tab:red') # type: ignore

        # Note, for station and sim station to line-up set the pre_trigger_time to 0 ns.

        for ax in axs:
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Voltage [V]")
            ax.legend(loc=4)

        plt.tight_layout()

        print(f'Saving trace for particle {iP} fields, detected at Station {station.get_id()}')
        fig.savefig(f"particle:{iP}_station:{station.get_id()}.png") # save the traces
        plt.close(fig)

