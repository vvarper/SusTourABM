import json
import os
import sys

import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts, dim

try:
    instance = sys.argv[1]
except IndexError:
    print(
        "{0} <instance>".format(sys.argv[0]))
    sys.exit(1)

instance_path = 'data/instances/' + instance + '.json'
instance_name = os.path.split(instance)[1].replace('_instance', '')
history_path = 'data/base_data/history_arrivals.csv'
experiment_folder = 'data/results/instances_output/'
plot_folder = experiment_folder + 'plots/'

results_file = experiment_folder + 'summary_outputs/' + instance_name + '_sumout.json'
flow_file = experiment_folder + 'summary_outputs/' + instance_name + '_sumflow2020to2050.csv'

# Load instance data (destinations) ###########################################
with open(instance_path) as json_file:
    instance_data = json.load(json_file)
    destinations = instance_data['destinations']
    num_destinations = len(destinations)

# Load (or calculate) flow data ###############################################

if not os.path.exists(flow_file):
    raw_path = experiment_folder + 'raw_outputs/' + instance_name + '_rawout.json'
    agents_path = experiment_folder + 'raw_outputs_with_agents/' + instance_name + '_rawout_agents.json'
    raw_flow_file = experiment_folder + 'raw_outputs/' + instance_name + '_rawflow2020to2050.csv'
    flow_matrix_file = experiment_folder + 'summary_outputs/' + instance_name + '_flowmatrix2020to2050.csv'

    # Crear los datos y guardar en flow_data
    with open(raw_path) as json_file:
        raw_data = json.load(json_file)
        raw_data = pd.DataFrame.from_dict(raw_data['results'])

    seeds = raw_data['seed'].unique()
    steps = raw_data['Step'].unique()

    with open(agents_path) as json_file:
        agent_data = json.load(json_file)
        agent_data = pd.DataFrame.from_dict(agent_data['results'])

    print('Archivo cargado')

    # Crear una matriz para almacenar los resultados de conteo
    counts = np.zeros((len(seeds), num_destinations, num_destinations))

    for seed in seeds:
        base_filter = (agent_data['seed'] == seed)
        filter_1 = base_filter & (agent_data['Step'] == 20)
        filter_2 = base_filter & (agent_data['Step'] == 49)

        pairs = agent_data[filter_1].merge(agent_data[filter_2],
                                           on=['seed', 'AgentID'],
                                           suffixes=('_1', '_2'))

        if not pairs.empty:
            counts[seed] = pd.crosstab(pairs['Choice_1'], pairs['Choice_2'],
                                       dropna=False).values

    column_names = [str(origin) + 'to' + str(destination) for origin in
                    range(num_destinations) for destination in
                    range(num_destinations)]

    df = pd.DataFrame(counts.reshape(-1, num_destinations ** 2),
                      columns=column_names)

    print('Matriz de conteo creada y rellena')

    # Insert seed column in the left
    df.insert(0, 'seed', seeds)
    df.to_csv(raw_flow_file, index=False)

    print('Matriz de conteo guardada')

    summary_df = df.drop(columns='seed').agg(['mean', 'std', 'min', 'max'])
    summary_df.to_csv(flow_file, index=True)

    # Build matrix of flows
    flow_matrix = np.zeros((num_destinations, num_destinations))

    for i in range(num_destinations):
        for j in range(num_destinations):
            flow_matrix[i, j] = summary_df.loc['mean'][str(i) + 'to' + str(j)]

    # Convert matrix to dataframe
    flow_matrix_df = pd.DataFrame(flow_matrix, index=destinations,
                                  columns=destinations)
    flow_matrix_df.to_csv(flow_matrix_file, index=True)

# Build chord diagram #########################################################

with open(flow_file) as f:
    flow_data = pd.read_csv(f, index_col=0)

nodes = hv.Dataset(pd.DataFrame.from_records(
    [{'index': i, 'name': destinations[i]} for i in range(len(destinations))]),
    'index')

# Create Pandas dataframe from an example list of dicts.
links = pd.DataFrame.from_records([{'source': i, 'target': j, 'value': round(
    flow_data.loc['mean'][str(i) + 'to' + str(j)])} for i in range(11) for j in
                                   range(11) if i != j])

hv.extension('bokeh')
hv.output(size=200)
nodes.data['group'] = 1

chord = hv.Chord((links, nodes))

chord.opts(opts.Chord(cmap='Category20', edge_cmap='Category20',
                      edge_color=dim('source').str(), labels='name',
                      node_color=dim('index').str(),
                      label_text_font_style='bold',
                      title='Flow from 2020 to 2050'))

# Save plot
output_file = plot_folder + 'chord_2020to2050_' + instance_name + '.html'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
hv.save(chord, output_file)
