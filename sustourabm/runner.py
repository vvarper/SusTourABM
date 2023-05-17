from mesa import batch_run

from sustourabm.model import SustainableTourismModel
from sustourabm.util.statistics import SimulationStatistics


def simulate(parameters, mc, num_processes, track_agents=False):
    parameters['seed'] = range(mc)
    parameters['track_agents'] = track_agents

    results = SimulationStatistics(
        batch_run(SustainableTourismModel, parameters=parameters,
                  data_collection_period=1, number_processes=num_processes,
                  iterations=1),
        num_destinations=parameters['num_destinations'],
        track_agents=parameters['track_agents'])

    return results
