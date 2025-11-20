import NuRadioReco.modules.io.eventReader
import inspect
import re
import os

def get_album_name():
    album_name = 'RNO_album_11_03_2025_20k_unorm'

    return album_name

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
        event.set_id(iP) # Sets the id of the event to the id of the particle

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

def get_rel_dir():
    """
    Returns the relative path from the current working directory to the directory containing the calling script.
    """
    # Get the filename of the caller
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    abs_file_path = os.path.abspath(caller_file)
    dir_name = os.path.dirname(abs_file_path)
    rel_path = os.path.relpath(dir_name, os.getcwd())
    return rel_path

def find_max_file_index(path):
    """
    Finds the maximum numeric index from all filenames in the given directory.
    Returns the highest number found across all filenames, or 0 if no numbers exist.
    """
    try:
        files = os.listdir(path)
    except (FileNotFoundError, PermissionError):
        return 0
    
    max_index = 0
    
    for filename in files:
        # Extract all digits from the filename (without path/extension)
        name = os.path.splitext(filename)[0]
        digits = re.findall(r'\d+', name)  # Find sequences of digits, not individual digits
        
        # Convert each digit sequence to int and find the max
        if digits:
            file_max = max(int(d) for d in digits)
            max_index = max(max_index, file_max)
    
    return max_index

def find_max_hdf5_index(hdf5_file):
    
    max_index = 0

    for keys in hdf5_file.keys():
        digits = re.findall(r'\d+', keys)

        if digits:
            file_max = max(int(d) for d in digits)
            max_index = max(max_index, file_max)

    return max_index