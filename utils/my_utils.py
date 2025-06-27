import NuRadioReco.modules.io.eventReader

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