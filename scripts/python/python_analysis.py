import multiprocessing as mp
import numpy as np
import pandas as pd
import time

from math import radians, cos, sin, asin, sqrt, pi
from pathlib import Path
from pathos.multiprocessing import ProcessPool as Pool
from sklearn.base import TransformerMixin, BaseEstimator
from toolz.sandbox.parallel import fold


def _haversine_dist(lat1, lon1, lat2, lon2):
    """

    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :return:
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    R = 6371  # Radius of earth in kilometers
    return c * R


def _calculate_poi_stats_df(poi_df):
    """

    :param poi_df:
    :return:
    """
    group_cols = ['POIID']

    poi_stats_df = poi_df \
        .loc[:, ['POIID', 'Distance']] \
        .groupby(group_cols) \
        .agg(['mean', 'std', 'max', 'count'])

    poi_stats_df.columns = ['mean_distance', 'stddev_distance', 'radius', 'req_count']
    poi_stats_df['density'] = poi_stats_df['req_count'] / (pi * pow(poi_stats_df['radius'], 2))

    return poi_stats_df


class DataPreprocessor:
    """

    """
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = PROJECT_DIR / 'data' / 'DataSample.csv'
    POI_PATH = PROJECT_DIR / 'data' / 'POIList.csv'

    def __init__(self):
        self.data_df = pd.read_csv(
            DataPreprocessor.DATA_PATH,
            names=['ID', 'TimeSt', 'Country', 'Province', 'City', 'Latitude', 'Longitude'],
            parse_dates=['TimeSt'],
            header=0,
            na_values=["\\N"]
        )

        self.poi_df = pd.read_csv(
            DataPreprocessor.POI_PATH,
            names=['POIID', 'Latitude', 'Longitude'],
            header=0,
            na_values=["\\N"]
        )

        self.__clean_data()

    def __clean_data(self):
        self.data_df \
            .drop_duplicates(subset=['TimeSt', 'Latitude', 'Longitude'], keep=False, inplace=True)

        self.poi_df \
            .drop_duplicates(subset=['Latitude', 'Longitude'], keep='first') \
            .reset_index(drop=True, inplace=True)


class POIAssigner(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, poi_df):
        self.poi_df = poi_df

    def __find_poi_id_and_distance(self, req_id, lat, long):
        distance_df = self.poi_df.apply(
            lambda row: _haversine_dist(lat, long, row['Latitude'], row['Longitude']),
            axis=1
        )

        poiid_col_index = self.poi_df.columns.get_loc('POIID')

        return pd.Series(
            [
                req_id,
                self.poi_df.iloc[distance_df.idxmin(), poiid_col_index],
                distance_df.iloc[distance_df.idxmin()]
            ],
            index=['ID', 'POIID', 'Distance']
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """

        :param X:
        :return:
        """
        x_copy = X.copy()

        result_df = x_copy.apply(
            lambda row: self.__find_poi_id_and_distance(row['ID'], row['Latitude'], row['Longitude']),
            result_type='expand',
            axis=1
        )

        return result_df


class POIAssignerMapReduce(POIAssigner):
    """

    """
    def __init__(self, poi_df, n_jobs=1):
        self.n_jobs = n_jobs

        MAX_WORKERS = mp.cpu_count()

        if self.n_jobs <= -1:
            self.num_partitions = MAX_WORKERS
        elif self.n_jobs == 0:
            self.num_partitions = 1
        else:
            self.num_partitions = min(self.n_jobs, MAX_WORKERS)

        super().__init__(poi_df)

    def __find_poi_id_and_distance(self, ID, lat, long):
        distance_df = self.poi_df.apply(
            lambda row: _haversine_dist(lat, long, row['Latitude'], row['Longitude']),
            axis=1
        )

        poiid_col_index = self.poi_df.columns.get_loc('POIID')

        return ID, self.poi_df.iloc[distance_df.idxmin(), poiid_col_index], distance_df.iloc[distance_df.idxmin()]

    def __transform_part(self, df_part):
        df_list = []

        for ID, lat, long in zip(df_part['ID'], df_part['Latitude'], df_part['Longitude']):
            df_list.append(self.__find_poi_id_and_distance(ID, lat, long))

        return pd.DataFrame(df_list, columns=['ID', 'POIID', 'Distance'])

    @staticmethod
    def __concat_df(df1, df2):
        """

        :param df1:
        :param df2:
        :return:
        """
        return pd.concat((df1, df2), axis="rows")

    def transform(self, X):
        """

        :param X:
        :return:
        """
        x_copy = X.copy()

        # splitting data into batches
        data_split_list = np.array_split(x_copy, self.num_partitions)

        with Pool(processes=self.num_partitions) as P:
            result_df = fold(
                self.__concat_df,
                P.imap(self.__transform_part, data_split_list),
                map=P.imap,
                combine=self.__concat_df
            )

        return result_df


if __name__ == '__main__':
    OUTPUT_DIR = DataPreprocessor.PROJECT_DIR / 'output'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preprocessed_data = DataPreprocessor()

    slow_poi_assignment = POIAssigner(preprocessed_data.poi_df)

    N_JOBS = 2
    fast_poi_assignment = POIAssignerMapReduce(preprocessed_data.poi_df, n_jobs=N_JOBS)

    start = time.process_time()
    output_df = slow_poi_assignment.transform(preprocessed_data.data_df)
    output_df.to_csv(OUTPUT_DIR / 'output.csv', index=False)
    print('Time elapsed for the iterative Pandas process: {} seconds'.format(time.process_time() - start))

    start = time.process_time()
    output_df = fast_poi_assignment.transform(preprocessed_data.data_df)
    output_df.to_csv(OUTPUT_DIR / 'output_via_mapreduce.csv', index=False)
    print('Time elapsed for the Pandas process using parallel map and reduce'
          ' with {} parallel processes: {} seconds'.format(N_JOBS, time.process_time() - start)

    poi_stats_df = _calculate_poi_stats_df(output_df)
    output_df.to_csv(OUTPUT_DIR / 'poi_stats.csv', index=False)



