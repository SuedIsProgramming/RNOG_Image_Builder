import NuRadioReco.modules.io.eventReader
from matplotlib import pyplot as plt
# from utils.my_utils import get_unique_events

plt.style.use('dark_background')

def get_unique_events(fPath):
    """
    Reads a NuRadioReco .nur file and returns a list of unique events, 
    where each event contains all stations that detected the same primary particle.

    This function iterates through all events in the file, groups together 
    stations that correspond to the same primary particle (using the primary particle ID), 
    and constructs a list of unique events, each with all associated stations.

    Parameters
    ----------
    fPath : str
        Path to the .nur file to be read.

    Returns
    -------
    events_unique : list of NuRadioReco.framework.event.Event
        List of unique events, each containing all stations that saw the same primary particle.

    Notes
    -----
    - The function assumes that the eventReader yields events in order of primary particle ID.
    - For each new primary particle ID, a new event is started and stations are added to it.
    - The function will output events with event ID equal to the particle ID they correspond to.
    """
    event_reader = NuRadioReco.modules.io.eventReader.eventReader()
    event_reader.begin(fPath)
    events_unique = []
    iFirst = -1
    first_event = None
    for iE, event in enumerate(event_reader.run()):
        primary = event.get_primary()
        iP = primary.get_id()
        event.set_id(iP)

        # Start a new unique first event if the primary ID changes
        if iFirst != event.get_id():
            if first_event is not None:
                events_unique.append(first_event)
            iFirst = event.get_id()
            first_event = event
        # Add stations from the same primary to the current first event
        if event is not first_event:    
            first_event.set_station(event.get_station())

    return events_unique

events_unique = get_unique_events('output.nur')

if len(events_unique) == 0:
    print('No events detected.')

for iE, event in enumerate(events_unique):
    if iE != 0:
        break
    stations = list(event.get_stations())
    stations_num = len(stations)
    # Get the number of channels for each station (assuming all stations have the same number of channels)
    channels_per_station = [len(list(station.iter_channels())) for station in stations]
    total_channels = sum(channels_per_station)

    fig, axs = plt.subplots(total_channels, 1, figsize=(12, 3 * total_channels), sharex=True, constrained_layout=True)
    if total_channels == 1:
        axs = [axs]

    ax_idx = 0
    for iStation, station in enumerate(stations):
        for ch in station.iter_channels():
            volts = ch.get_trace()
            times = ch.get_times()
            tot_time = int(times[-1] - times[0]) + 1
            axs[ax_idx].plot(times, volts, label=f'Station {station.get_id()} Channel {ch.get_id()}', linewidth=1.5)
            axs[ax_idx].set_title(
                rf"Event {event.get_id()} | Station {station.get_id()} | Channel {ch.get_id()} | $t_{{trigger}} = {tot_time}$ ns",
                fontsize=11, fontweight='bold'
            )
            axs[ax_idx].set_ylabel("Voltage [V]", fontsize=10)
            axs[ax_idx].legend(loc=4, fontsize=9)
            axs[ax_idx].grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
            ax_idx += 1

    axs[-1].set_xlabel("Time [ns]", fontsize=11)
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.set_facecolor('#f7f7f7')

    station_ids = [station.get_id() for station in stations]
    print(f'Saving trace for particle {event.get_id()}, detected at Stations: {station_ids}')
    fig.suptitle(f"Event {event.get_id()} | Detected at {stations_num} Stations", fontsize=14, fontweight='bold')
    fig.savefig(f"particle_{event.get_id()}_detected_at_{stations_num}_stations.png", dpi=150)
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

