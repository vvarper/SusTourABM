"""Main module of the agent-based model for sustainable tourism.
"""

import logging
from itertools import compress
from typing import Optional

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from scipy.special import softmax

log = logging.getLogger(__name__)


class Destination:
    """Class for the tourist destinations.

    The state of a destination is determined by 10 factors:
        1. infectious_diseases
        2. heat_waves
        3. beach_loss
        4. water_shortages
        5. forest_fires
        6. marine_habitats
        7. terrestrial_habitats
        8. infrastructures
        9. cultural_heritage
        10. price

    Attributes
    ----------
    unique_id : int
        Unique identifier for the destination.
    states : np.ndarray
        Array containing the state values at each step for each factor.
    availability : float
        Probability of the destination being available at any time-step
    """

    def __init__(self, unique_id: int, attributes: np.ndarray,
                 availability: float):
        self.unique_id: int = unique_id
        self.states: np.ndarray = np.copy(attributes)
        self.availability: float = availability


class SustainableTourismModel(Model):
    """ Agent-based model for a sustainable tourism scenario.

    Attributes
    ----------
    num_destinations: int
        Number of tourist destinations.
    num_tourists: int
        Number of tourists agents.
    num_steps: int
        Number of simulation steps.
    current_step: int
        Current simulation step.
    destinations : list[Destination]
        List of tourist destinations.
    schedule : RandomActivation
        Scheduler of the tourist agents
    running: bool
        Flag indicating if the simulation is running.
    datacollector: DataCollector
        Data collector for the simulation.
    """

    def __init__(
            self,
            state_by_destination_step_factor: tuple[tuple[
                tuple[float]]],  # float[num_destinations][num_steps][10]
            mean_tourist_preferences_by_factor: tuple[float],  # float[10]
            tourist_preferences_deviation: float,
            availability_by_destination: tuple[float],
            num_destinations: int = 11,  # > 0
            num_tourists: int = 10152,  # > 0
            num_steps: int = 240,  # > 0
            seed: Optional[int] = None,
            track_agents: bool = False):
        """Initialize the model.

        Parameters
        ----------
        state_by_destination_step_factor : tuple[tuple[tuple[float]]]
            State values for each destination by step and factor.
        mean_tourist_preferences_by_factor : tuple[float]
            Mean values for the tourist preferences by factor.
        tourist_preferences_deviation : float
            Standard deviation of the tourist preferences.
        availability_by_destination : tuple[float]
            Availability value for each destination.
        num_destinations: int
            Number of tourist destinations.
        num_tourists: int
            Number of tourists agents.
        num_steps: int
            Number of simulation steps.
        seed: Optional[int]
            Random seed for the random number generator.
        """

        # Initialize model attributes
        self.num_destinations: int = num_destinations
        self.num_tourists: int = num_tourists
        self.num_steps: int = num_steps
        self.current_step: int = 0
        self.random = np.random.default_rng(seed)

        # Create destinations
        self.destinations: list[Destination] = []
        self.__create_destinations(state_by_destination_step_factor,
                                   availability_by_destination)

        # Create tourists' scheduler with two stages
        self.schedule: RandomActivation = RandomActivation(self)

        # Create tourists
        self.__create_tourist_agents(mean_tourist_preferences_by_factor,
                                     tourist_preferences_deviation)

        # Create destination reporters
        destination_reporters: dict = {
            'Arrivals ' + str(destination.unique_id):
                [self.calculate_current_num_tourists_in_destination,
                 [destination]]
            for destination in self.destinations
        }

        # Create data collectors
        agent_reporters = {
            'Choice': 'current_destination'
        } if track_agents else None

        self.datacollector: DataCollector = DataCollector(
            model_reporters=destination_reporters,
            agent_reporters=agent_reporters)

        # Start simulation
        self.running: bool = True

    def __create_destinations(
            self, state_by_destination_step_factor: tuple[tuple[tuple[float]]],
            availability_by_destination: tuple[float]
    ) -> None:
        """Create the tourist destinations.

        Side effects
        -----------
        The attribute 'destinations' is filled with the created destinations.

        Parameters
        ----------
        state_by_destination_step_factor : tuple[tuple[tuple[float]]]
            State values for each destination by step and factor.
        """

        for i, state in enumerate(np.array(state_by_destination_step_factor)):
            self.destinations.append(
                Destination(i, state, availability_by_destination[i]))

    def __create_tourist_agents(
            self, mean_tourist_preferences_by_factor: tuple[float],
            tourist_preferences_deviation: float) -> None:
        """Create the tourist agents and add them to the scheduler.

        Side effects:
        ----------
        The attribute 'schedule' is filled.

        Parameters
        ----------
        mean_tourist_preferences_by_factor : tuple[float]
            Mean values for the tourist preferences by factor.
        tourist_preferences_deviation : float
            Standard deviation of the tourist preferences.
        """

        for i in range(self.num_tourists):
            preferences: np.ndarray = np.clip(self.random.normal(
                loc=mean_tourist_preferences_by_factor,
                scale=tourist_preferences_deviation),
                a_min=None,
                a_max=0)

            tourist = Tourist(i, self, preferences)
            self.schedule.add(tourist)

    def calculate_current_num_tourists_in_destination(
            self, destination: Destination) -> int:
        """Return the current number of tourists in the given destination.

        Parameters
        ----------
        destination : Destination
            Destination of interest.

        Returns
        -------
        int
            Current number of tourists in the destination.
        """

        num_tourists = sum([
            tourist.current_destination == destination.unique_id
            for tourist in self.schedule.agents
        ])

        log.debug(
            f'Destination {destination.unique_id} has {num_tourists} tourists')

        return num_tourists

    def finish(self) -> bool:
        """Indicate if the simulation is finished.

        Returns
        -------
        bool
            True if the simulation is finished, False otherwise.
        """
        return self.current_step == self.num_steps

    def step(self) -> None:
        """Run a simulation step

        Side effects:
        -------------
        The current step is incremented.
        All agents perform their actions and change their states.
        The data collector is updated.
        The simulation is finished if the final step is reached.
        """

        self.schedule.step()
        self.current_step += 1
        self.datacollector.collect(self)

        if self.finish():
            self.running = False

    def run_model(self) -> None:
        """Run a simulation to the end.

        Side effects:
        -------------
        The current step is incremented to the final step.
        All agents perform their actions and change their states.
        The data collector is updated.
        The simulation is finished.
        """
        while self.running:
            self.step()


class Tourist(Agent):
    """Class representing a tourist agent.

    An agent tourist travels to a destination after choosing it.
    -1 value for the destination means that the tourist is not traveling.

    Attributes
    ----------
    unique_id : int
        Unique identifier for the tourist.
    model : SustainableTourismModel
        Model where the tourist operates.
    current_destination : int
        Destination of the tourist in the current step.
    preferences : np.ndarray
        Preference values of the tourist for each factor.
    """

    def __init__(self, unique_id: int, model: SustainableTourismModel,
                 preferences: np.ndarray):  # float[10]
        super().__init__(unique_id, model)

        self.current_destination: int = -1
        self.preferences = preferences / preferences.sum()

    def calculate_preference(self, destination: Destination) -> float:
        """Calculate the preference value of the tourist for the given 
        destination.

        Parameters
        ----------
        destination : Destination
            Destination of interest.

        Returns
        -------
        float
            Preference value of the tourist for the given destination.
        """

        return -self.preferences.dot(
            destination.states[self.model.current_step])

    def step(self) -> None:
        """Perform the decision-making process of the tourist.

        Side effects:
        -------------
        The current destination is updated.
        """

        # Determine which destinations are available
        destinations = self.model.destinations

        is_available = []
        while not any(is_available):
            is_available = [
                self.random.random() < destination.availability
                for destination in destinations]

        available_destinations = list(
            compress(destinations, is_available))

        # Destination scores
        destination_scores = np.array([
            self.calculate_preference(destination)
            for destination in available_destinations
        ])

        # Calculate travel probabilities and make decision
        travel_probabilities = softmax(destination_scores)

        self.current_destination = self.random.choice(
            a=available_destinations, p=travel_probabilities).unique_id
