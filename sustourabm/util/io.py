"""Module with general purposed utility functions.
"""
import json
import os

import pandas as pd


###############################################################################
# GENERAL FUNCTIONS


def totuple(list: list):
    try:
        return tuple(totuple(i) for i in list)
    except TypeError:
        return list


###############################################################################
# JSON-PANDAS FUNCTIONS

def load_model_instance_from_json(input_filename):
    with open(input_filename) as f:
        data = json.load(f)

    parameters = data["abm_parameters"]
    climate_factors = data["climate_factors"]
    destinations = data["destinations"]

    for key in parameters:
        if type(parameters[key]) == list:
            parameters[key] = [totuple(parameters[key])]

    return parameters, climate_factors, destinations


def load_history_from_csv(filename):
    history = pd.read_csv(filename, index_col=0)
    share_columns = [title for title in history.columns if
                     title.startswith('Share')]

    history = history[['Year'] + share_columns]

    return history


def save_dict2json(data, output_filename, indent=None, sort_keys=False):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)


def save_model_instance(output_filename, climate_factors, destinations,
                        abm_parameters):
    instance_data = {"climate_factors": climate_factors,
                     "destinations": destinations,
                     "abm_parameters": abm_parameters}

    save_dict2json(instance_data, output_filename, indent=4, sort_keys=True)


def save_simulation_output(output_filename, instance_name, parameters,
                           results):
    output_data = {'instance_name': instance_name, 'parameters': parameters,
                   'results': dict((str(k), v) for k, v in results.items())}

    save_dict2json(output_data, output_filename)
