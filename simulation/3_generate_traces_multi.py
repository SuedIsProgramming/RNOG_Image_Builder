from utils.my_utils import get_unique_events, get_rel_dir
from matplotlib import pyplot as plt

#plt.style.use('dark_background')

rel_dir = get_rel_dir() 

events_unique = get_unique_events(f'{rel_dir}/output.nur')

if len(events_unique) == 0:
    print('No events detected.')

for iE, event in enumerate(events_unique):
    stations = list(event.get_stations())
    stations_num = len(stations)
    # Get the number of channels for each station (assuming all stations have the same number of channels)
    channels_per_station = [len(list(station.iter_channels())) for station in stations]
    total_channels = sum(channels_per_station)

    fig, axs = plt.subplots(total_channels, 1, figsize=(12, 3 * total_channels), sharex=True, constrained_layout=True)
    if total_channels == 1:
        axs = [axs]

    ax_idx = 0
    
    max_volt = max( 
        max(channel.get_hilbert_envelope())
        for station in stations
        for channel in station.iter_channels()
    )

    for iStation, station in enumerate(stations):
        for ch in station.iter_channels():
            volts = ch.get_hilbert_envelope()
            times = ch.get_times()
            tot_time = int(times[-1] - times[0]) + 1
            axs[ax_idx].plot(times, volts, label=f'Station {station.get_id()} Channel {ch.get_id()}', linewidth=1.5) # type:ignore
            axs[ax_idx].set_title( # type:ignore
                rf"Event {event.get_id()} | Station {station.get_id()} | Channel {ch.get_id()} | $t_{{trigger}} = {tot_time}$ ns",
                fontsize=11, fontweight='bold'
            )
            axs[ax_idx].set_ylabel("Voltage [V]", fontsize=10) # type:ignore
            axs[ax_idx].set_ylim(0, max_volt) # type:ignore
            axs[ax_idx].legend(loc=4, fontsize=9) # type:ignore
            axs[ax_idx].grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7) # type:ignore
            ax_idx += 1

    axs[-1].set_xlabel("Time [ns]", fontsize=11) # type:ignore
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=9) # type:ignore
        ax.set_facecolor('#f7f7f7') # type:ignore

    station_ids = [station.get_id() for station in stations]

    fpath = f'images_and_traces/traces_particle_{event.get_id()}_detected_at_{stations_num}.png'

    print(f'Saving {fpath}...')
    fig.suptitle(f"Event {event.get_id()} | Detected at {stations_num} Stations", fontsize=14, fontweight='bold')
    fig.savefig(fpath, dpi=150)
    plt.close(fig)

################################## OLD ###################################################

# from matplotlib import pyplot as plt
# from matplotlib.offsetbox import AnchoredText

# import NuRadioReco.modules.io.eventReader
# event_reader = NuRadioReco.modules.io.eventReader.eventReader()

# file = 'output.nur'
# event_reader.begin(file)
# for iE, event in enumerate(event_reader.run()):
#     primary = event.get_primary()
#     iP = primary.get_id()

#     for iStation, station in enumerate(event.get_stations()):
#         stations = list(event.get_stations())
#         stations_num = len(stations)
#         channels = list(station.iter_channels())
#         channels_num = len(channels)

#         # a fig and axes for our waveforms
#         fig, axs = plt.subplots(stations_num*channels_num, 1, figsize=(5,20)) # subplots will scale with num of total channels

#         # this loops through "mock data" (with noise added, etc.)
#         for ch in station.iter_channels():
#             volts = ch.get_trace()
#             times = ch.get_times()
#             tot_time = (times[-1] - times[0]) + 1 # Have to add one ns because first element starts at 1 ns.
#             tot_time_box = AnchoredText(f'$t_{{tot}}:$ {int(tot_time)} ns',loc=1)
#             axs[ch.get_id()].add_artist(tot_time_box) # Add total time of trace (Only calculated for true V, but should be the same regardless)
#             axs[ch.get_id()].plot(times, volts, label='V') # type: ignore
#             axs[ch.get_id()].set_title(f"Particle number {iP} | Station {station.get_id()}, Channel {ch.get_id()}") # type: ignore
        
#         # this loops through *MC truth* waveforms (before noise was added)
#         # this may prove useful at some point
#         if station.has_sim_station():
#             sim_station = station.get_sim_station()
#             for sim_ch in sim_station.iter_channels():
#                 volts = sim_ch.get_trace()
#                 times = sim_ch.get_times()
#                 if sim_ch.get_ray_tracing_solution_id() == 0: # If else check to add direct and reflected traces.
#                     axs[sim_ch.get_id()].plot(times, volts, '--',label=f'Vsim direct',color='tab:green') # type: ignore
#                 else:
#                     axs[sim_ch.get_id()].plot(times, volts, '--',label=f'Vsim reflected',color='tab:red') # type: ignore

#         # Note, for station and sim station to line-up set the pre_trigger_time to 0 ns.

#         for ax in axs:
#             ax.set_xlabel("Time [ns]")
#             ax.set_ylabel("Voltage [V]")
#             ax.legend(loc=4)

#         plt.tight_layout()

#         print(f'Saving trace for particle {iP} fields, detected at Station {station.get_id()}')
#         fig.savefig(f"particle:{iP}_station:{station.get_id()}.png") # save the traces
#         plt.close(fig)

