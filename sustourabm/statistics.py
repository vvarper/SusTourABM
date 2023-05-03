import pandas as pd


class SimulationStatistics:
    def __init__(self, results, num_destinations):
        self.raw_results = self.__calculate_raw_results(results,
                                                        num_destinations)
        self.summary_results = self.__summarise_results(self.raw_results,
                                                        'Step')

    @staticmethod
    def __calculate_raw_results(results, num_destinations):
        arrival_names = ['Arrivals ' + str(i) for i in range(num_destinations)]
        column_names = ['seed', 'Step'] + arrival_names
        df = pd.DataFrame(results)[column_names].sort_values(
            by=['seed', 'Step'])
        num_tourists = df[arrival_names].sum(axis=1)
        df.insert(2, 'Num tourists', num_tourists)
        for dest in range(num_destinations):
            df['Share ' + str(dest)] = df[arrival_names[dest]] / df[
                'Num tourists']

        return df

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
