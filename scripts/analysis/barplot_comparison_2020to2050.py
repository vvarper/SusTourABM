import json
import os
import re
import sys

import pandas as pd
from matplotlib import pyplot as plt


def convert_df_to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    columns = [re.sub(r'\'|\(|\)', '', column).split(', ') for column in
               df.columns]

    second_level_size = len(set([column[1] for column in columns]))
    first_level = [column[0] for column in columns[::second_level_size]]
    second_level = [column[1] for column in columns[:second_level_size]]
    multi_index = pd.MultiIndex.from_product([first_level, second_level])
    df.columns = multi_index

    return df


try:
    instance = sys.argv[1]
except IndexError:
    print("{0} <instance>".format(sys.argv[0]))
    sys.exit(1)

instance_path = f'data/instances/{instance}.json'
instance_name = os.path.split(instance)[1].replace('_instance', '')
plot_folder = 'data/results/instances_output/plots/'
output_file = f'data/results/instances_output/summary_outputs/' \
              f'{instance_name}_sumout.json'

# Load instance data (destinations) ###########################################
with open(instance_path) as json_file:
    instance_data = json.load(json_file)
    destinations = instance_data['destinations']
    num_destinations = len(destinations)

# Load results data ###########################################################
with open(output_file) as json_file:
    data = json.load(json_file)
    results = pd.DataFrame.from_dict(data['results'])
    results = convert_df_to_multiindex(results)

    share_columns = [column for column in results.columns.levels[0] if
                     column.startswith('Share')]
    results = results[results.columns[
        results.columns.get_level_values(0).isin(share_columns)]]
    results = results[
        results.columns[results.columns.get_level_values(1) == 'mean']]
    results = results.iloc[[20, 49]]

    results.columns = destinations
    results.index = ['2020', '2049']

fig = results.transpose().plot.bar(rot=0, figsize=(10, 5))
fig.set_ylabel('Share')
fig.set_xlabel('Destination')
fig.set_title(f'Share per destination in {instance_name} configuration\n')

# Save figure
figure_path = f'{plot_folder}{instance_name}_2020to2050_barplot_comparison.png'
os.makedirs(os.path.dirname(figure_path), exist_ok=True)
plt.savefig(figure_path, bbox_inches='tight')

plt.clf()

difference = results.loc['2049'] - results.loc['2020']

fig = difference.plot.bar(rot=0, figsize=(10, 5), width=0.9,
                          color=['red' if val < 0 else 'blue' for val in
                                 difference])
fig.set_ylabel('Share difference')
fig.set_xlabel('Destination')
fig.set_title(
    f'Share difference per destination between 2020 and 2049 in '
    f'{instance_name} configuration\n')

for i, val in enumerate(difference):
    if val >= 0:
        fig.text(i, val, str(round(val, 4)), ha='center', va='bottom')
    else:
        fig.text(i, val, str(round(val, 4)), ha='center', va='top')

fig.axhline(linewidth=1, color='black')

# Save figure
figure_path = f'{plot_folder}{instance_name}_2020to2050_barplot_diff.png'
os.makedirs(os.path.dirname(figure_path), exist_ok=True)
plt.savefig(figure_path, bbox_inches='tight')
