from matplotlib import pyplot as plt
import NuRadioReco.modules.io.eventReader

event_reader = NuRadioReco.modules.io.eventReader.eventReader()

file = 'output.nur'
event_reader.begin(file)

events_unique = []
iFirst = -1
first_event = None
for event in event_reader.run():
    primary = event.get_primary()
    iP = primary.get_id()
    event.set_id(iP)
    if iFirst is not event.get_id():
        if first_event is not None:
            events_unique.append(first_event)
        iFirst = event.get_id()
        first_event = event
    if event is not first_event:    
        first_event.set_station(event.get_station())
    

