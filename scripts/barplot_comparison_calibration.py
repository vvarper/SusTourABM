import json
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


def get_experiment_total_share(output_file, destinations):
    with open(output_file) as json_file:
        data = json.load(json_file)

        # Read results from json
        year_rawout = pd.DataFrame.from_dict(data['results'])
        arrival_columns = [title for title in year_rawout.columns if
                           title.startswith('Arrivals')]

        # Select years 2010-2019
        year_rawout = year_rawout[
            (year_rawout['Step'] >= 10) & (year_rawout['Step'] < 20)]

        # Group arrivals by seed (sum arrivals of all years by seed)
        year_rawout = year_rawout[arrival_columns + ['seed']].groupby(
            'seed').sum()
        year_rawout['Total'] = year_rawout.sum(axis=1)

        # Get share per seed and calculate mean for each destination
        total_shares = year_rawout[arrival_columns].divide(
            year_rawout['Total'],
            axis=0).mean()

        # Rename columns to destination names (readability)
        total_shares.rename(
            lambda name: destinations[int(re.findall(r'\d+', name)[0])],
            inplace=True)

        return total_shares


try:
    rcp = sys.argv[1]
except IndexError:
    print("{0} <RCP>".format(sys.argv[0]))
    sys.exit(1)

# Paths / args
history_path = 'data/base_data/history_arrivals.csv'
plot_folder = 'data/results/instances_output/plots/'
solutions = [f'calibrated_{rcp}', rcp]

################################################################################

# 1. Load history data
history = pd.read_csv(history_path, index_col=0)

# Get arrival column names
arrival_columns = [title for title in history.columns if not title.startswith(
    'Share') and title != 'Year' and title != 'Total']

# Get total shares per destination
history_total_shares = history[arrival_columns].sum() / history['Total'].sum()
destinations = list(history_total_shares.index)

# 2. Get all experiment data
total_shares = pd.DataFrame(history_total_shares.rename('History'))
for solution in solutions:
    # Get experiment output file (year raw)
    output_file = f'data/results/instances_output/raw_outputs/{solution}_rawout.json'

    # Calculate total shares (given alpha) and join to dataframe
    config_share = get_experiment_total_share(output_file, destinations)
    total_shares = total_shares.join(config_share.rename(solution))

total_shares.columns = ['History', 'Calibrated Solution', 'Initial Solution']

# Build figure for given paramter config
fig = total_shares.plot.bar(rot=0, figsize=(10, 5))
fig.set_ylabel('Total share (2010-2019)')
fig.set_xlabel('Destination')
fig.set_title(f'Total share per destination\n')

# Save figure
figure_path = f'{plot_folder}{rcp}_calibration_barplot_comparison.png'
os.makedirs(os.path.dirname(figure_path), exist_ok=True)
plt.savefig(figure_path, bbox_inches='tight')
