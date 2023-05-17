import numpy as np


def define_factors_5_dest_states_map(climate_preferences, calibrate_states,
                                     calibrate_factors, variation_range):
    int_ranges = []
    real_ranges = []
    chromosome_map = []

    # Climate factors order: [0 4 6 7 8 5 3 1 2 9]
    climate_preferences_order = np.argsort(climate_preferences)

    # 2. Destination states
    if calibrate_states:
        # Map: ('state_by_destination_step_factor', destination, factor)
        # Destination order: Canary, Balearic, Sicily, Crete, Sardinia
        destination_order = [0, 1, 7, 2, 8]

        state_parameters_map = [
            ('state_by_destination_step_factor', dest_order, factor_order) for
            dest_order in destination_order for factor_order in
            climate_preferences_order]

        num_states_parameters = len(destination_order) * len(
            climate_preferences_order)

        chromosome_map += state_parameters_map
        int_ranges += [(0, 1000) for _ in range(num_states_parameters)]
        real_ranges += [(-variation_range, variation_range) for _ in
                        range(num_states_parameters)]

    # 3. Climate preferences
    if calibrate_factors:
        climate_parameters_map = [('mean_tourist_preferences_by_factor', order)
                                  for order in climate_preferences_order]

        chromosome_map += climate_parameters_map
        int_ranges += [(0, 1000) for _ in climate_preferences]
        real_ranges += [((1 - variation_range) * preference,
                         (1 + variation_range) * preference) for preference in
                        climate_preferences[climate_preferences_order]]

    real_ranges = np.array(real_ranges)
    int_ranges = np.array(int_ranges)

    return chromosome_map, real_ranges, int_ranges
