"""Module with general purposed utility functions.
"""
import json
import os


###############################################################################
# GENERAL FUNCTIONS


def totuple(list: list):
    try:
        return tuple(totuple(i) for i in list)
    except TypeError:
        return list


###############################################################################
# JSON-PANDAS FUNCTIONS

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
