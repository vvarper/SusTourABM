import numpy as np
import pandas as pd

from sustourabm.util.io import save_model_instance
from sustourabm.util.io import to_tuple

impact_ranges = {'Infectious diseases': [0, 100], 'Heat waves': [0, 365],
                 'Beaches availability': [0, 1], 'Water shortages': [0, 5],
                 'Forest fires': [0, 1], 'Marine habitats': [0, 1],
                 'Land habitats': [0, 1], 'Infrastructure damage': [0, 1],
                 'Cultural heritage damage': [0, 1], 'Attractiveness': [0, 1]}


###############################################################################


def scale_impact_factors(dataframe):
    scaled_dataframe = dataframe.copy()
    scaled_dataframe[['2000', '2050', '2100']] = scaled_dataframe.apply(
        lambda row: row[['2000', '2050', '2100']] /
                    impact_ranges[row['Impact']][1], axis=1)

    return scaled_dataframe


def create_destination_state_by_step_factor(dataframe: pd, destination: str):
    start = dataframe[dataframe['Destination'] == destination]['2000'].values
    stop = dataframe[dataframe['Destination'] == destination]['2050'].values

    return np.swapaxes(
        np.linspace(start=start, stop=stop, endpoint=True, axis=-1, num=50), 0,
        1)


def create_states_by_destination_step_factor(dataframe: pd.DataFrame,
                                             destinations):
    return np.array(
        [create_destination_state_by_step_factor(dataframe, destination) for
         destination in destinations])


###############################################################################

df_rcp26 = pd.read_csv('data/base_data/impacts_rcp26.csv')
df_rcp85 = pd.read_csv('data/base_data/impacts_rcp85.csv')

scaled_rcp26 = scale_impact_factors(df_rcp26)
scaled_rcp85 = scale_impact_factors(df_rcp85)

climate_factors = tuple(scaled_rcp26['Impact'].unique())
destinations = tuple(df_rcp26['Destination'].unique())

states_rcp26 = to_tuple(
    create_states_by_destination_step_factor(scaled_rcp26, destinations))
states_rcp85 = to_tuple(
    create_states_by_destination_step_factor(scaled_rcp85, destinations))

df_mean_tourist_preferences_by_factor = pd.read_csv(
    'data/base_data/tourist_preferences.csv')
assert climate_factors == tuple(df_mean_tourist_preferences_by_factor[
                                    'Impact']), 'Inconsistent climate factors'

df_availability_by_destination = pd.read_csv(
    'data/base_data/destination_availability.csv')
assert destinations == tuple(
    df_availability_by_destination['Destination']), 'Inconsistent destinations'

tourist_preferences_deviation = 0
num_destinations = len(destinations)
num_tourists = 10152
num_steps = len(states_rcp26[0])  # 50 years: 2000 to 2049

params = {"state_by_destination_step_factor": states_rcp26,
          "mean_tourist_preferences_by_factor": tuple(
              df_mean_tourist_preferences_by_factor['Preference']),
          "tourist_preferences_deviation": tourist_preferences_deviation,
          "availability_by_destination": tuple(
              df_availability_by_destination['Availability']),
          "num_destinations": num_destinations, "num_tourists": num_tourists,
          "num_steps": num_steps, }

rcp26_instance = {"climate_factors": climate_factors,
                  "destinations": destinations, "abm_parameters": params}

output_filename = 'data/instances/original/rcp26_instance.json'
save_model_instance(output_filename, climate_factors, destinations, params)

params['state_by_destination_step_factor'] = states_rcp85
output_filename = 'data/instances/original/rcp85_instance.json'
save_model_instance(output_filename, climate_factors, destinations, params)
