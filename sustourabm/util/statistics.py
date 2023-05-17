import numpy as np
import pandas as pd

from sustourabm.util.io import save_simulation_output


class SimulationStatistics:
    def __init__(self, results, num_destinations, track_agents=False):

        self.track_agents = track_agents
        self.raw_results = self.__calculate_raw_results(results,
                                                        num_destinations)
        self.summary_results = self.__summarise_results(self.raw_results,
                                                        'Step')

    def __calculate_raw_results(self, results, num_destinations):
        arrival_columns = ['Arrivals ' + str(i) for i in
                           range(num_destinations)]
        base_columns = ['seed', 'Step']
        general_columns = base_columns + arrival_columns

        if self.track_agents:
            agent_columns = base_columns + ['AgentID', 'Choice']
            agent_results = []
            general_results = []

            for result in results:
                if 'AgentID' in result:
                    if result['AgentID'] == 0:
                        general_results.append(
                            {key: result.get(key) for key in general_columns})
                    agent_results.append(
                        {key: result.get(key) for key in agent_columns})
                else:
                    general_results.append(
                        {key: result.get(key) for key in general_columns})

            general_results = pd.DataFrame(general_results).sort_values(
                by=['seed', 'Step']).reset_index(drop=True)

            agent_results = pd.DataFrame(agent_results).sort_values(
                by=['seed', 'Step', 'AgentID']).reset_index(drop=True)

            self.agent_results = agent_results

            seeds = general_results['seed'].unique()
            steps = general_results['Step'].unique()

            # Crear una matriz para almacenar los resultados de conteo
            counts = np.zeros(
                (len(steps) * len(seeds), num_destinations, num_destinations))

            # Iterar sobre los valores únicos de 'seed' y 'Step'
            for seed in seeds:
                base_filter = (agent_results['seed'] == seed)
                for step in steps[2:]:
                    filter_1 = base_filter & (
                            agent_results['Step'] == step - 1)
                    filter_2 = base_filter & (agent_results['Step'] == step)

                    pairs = agent_results[filter_1].merge(
                        agent_results[filter_2], on=['seed', 'AgentID'],
                        suffixes=('_1', '_2'))

                    if not pairs.empty:
                        counts_idx = seed * len(steps) + step - 1
                        counts[counts_idx] = pd.crosstab(pairs['Choice_1'],
                                                         pairs['Choice_2'],
                                                         dropna=False).values

            column_names = [str(origin) + 'to' + str(destination) for origin in
                            range(num_destinations) for destination in
                            range(num_destinations)]

            temp_df = pd.DataFrame(counts.reshape(-1, num_destinations ** 2),
                                   columns=column_names)
            general_results = pd.concat([general_results, temp_df], axis=1)

        else:
            general_results = pd.DataFrame(results)[
                general_columns].sort_values(by=['seed', 'Step'])
            self.agent_results = None

        num_tourists = general_results[arrival_columns].sum(axis=1)
        general_results.insert(2, 'Num tourists', num_tourists)
        for dest in range(num_destinations):
            general_results['Share ' + str(dest)] = general_results[
                                                        arrival_columns[
                                                            dest]] / \
                                                    general_results[
                                                        'Num tourists']

        return general_results

    @staticmethod
    def __summarise_results(results, column):
        summary_df = results.drop(columns='seed').groupby([column]).agg(
            ['mean', 'std', 'min', 'max'])

        df_mean = summary_df.mean()
        df_std = summary_df.std()
        df_min = summary_df.min()
        df_max = summary_df.max()

        summary_df.loc['mean'] = df_mean
        summary_df.loc['std. dev.'] = df_std
        summary_df.loc['min'] = df_min
        summary_df.loc['max'] = df_max

        return summary_df

    def save(self, base_output_folder, instance_name, simulation_config):

        if self.track_agents:
            raw_output_path = base_output_folder + 'raw_outputs_with_agents/' + instance_name + '_rawout.json'
            agents_output_path = base_output_folder + 'raw_outputs_with_agents/' + instance_name + '_rawout_agents.json'
            summary_output_path = base_output_folder + 'summary_outputs_with_agents/' + instance_name + '_sumout.json'
        else:
            raw_output_path = base_output_folder + 'raw_outputs/' + instance_name + '_rawout.json'
            summary_output_path = base_output_folder + 'summary_outputs/' + instance_name + '_sumout.json'

        save_simulation_output(raw_output_path, instance_name,
                               simulation_config,
                               self.raw_results.to_dict())
        save_simulation_output(summary_output_path, instance_name,
                               simulation_config,
                               self.summary_results.to_dict())

        if self.track_agents:
            save_simulation_output(agents_output_path, instance_name,
                                   simulation_config,
                                   self.agent_results.to_dict())
