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
    Calculate the distance between two points (say, A and B) from their geographical coordinates using the Haversine
    formula

    :param lat1: Latitude of point A
    :param lon1: Longitude of point A
    :param lat2: Latitude of point B
    :param lon2: Longitude of point B
    :return: A floating point number representing the distance in kilometres
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
    Calculate all the relevant statistics for a given position of interest (POI)

    :param poi_df: A Pandas dataframe containing information mapping a request ID to a POI together with the distance
    between the ID and the POI
    :return: A Pandas dataframe containing information about the mean and standard deviations of the distances of all
    various requests, the maximum distance (radius), request counts, and density associated to a POI
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
    Encapsulate the reading and pre-processing of all associated data for the project
    """
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = PROJECT_DIR / 'data' / 'DataSample.csv'
    POI_PATH = PROJECT_DIR / 'data' / 'POIList.csv'
    """ Paths for storing the location of all the relevant data"""

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
    Encapsulate operations responsible for mapping a request ID to its nearest POI

    Extend BaseEstimator and TransformerMixin from scikit-learn, for use in scitkit-learn pipelines, if necessary
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
        Apply the transformation process (done here iteratively) to a given Pandas dataframe

        :param X: A Pandas dataframe containing information about requests
        :return: A Pandas dataframe containing request ID to its nearest POI, together with distance between the two
        points
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
    Encapsulate operations responsible for mapping a request ID to its nearest POI while making use of parallel map
    and reduce operations in order to speed up the process considerably

    Extend BaseEstimator and TransformerMixin from scikit-learn, for use in scitkit-learn pipelines, if necessary
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
        Implement an associative operation that executes reductions in parallel

        The associative operation in this is case is concatenation. The output dataframes of the parallel map operation
        are concatenated row-wise to yield the resultant final dataframe. More information regarding this parallel map
        and reduction process can be found in the documentation for the fold function within the Pytoolz Python package:
        https://toolz.readthedocs.io/en/latest/api.html#toolz.sandbox.parallel.fold

        :param df1: A Pandas dataframe that is the output of a parallel map operation
        :param df2: A Pandas dataframe that is the output of a parallel map operation
        :return: A Pnadas dataframe formed by the row-wise concatenation of the input dataframes
        """
        return pd.concat((df1, df2), axis="rows")

    def transform(self, X):
        """
        Apply the transformation process (done via map and reduce) to a given Pandas dataframe

        The inital map operation is achieved by splitting the larger dataframe containing information about the requests
        into smaller batches. Each batch is then evaluated lazily within a parallel reduce operation, executed by the
        use of the fold function within the sandbox.parallel module of the Pytoolz package

        :param X: A Pandas dataframe containing information about requests
        :return: A Pandas dataframe containing request ID to its nearest POI, together with distance between the two
        points
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
          ' with {} parallel processes: {} seconds'.format(N_JOBS, time.process_time() - start))

    poi_stats_df = _calculate_poi_stats_df(output_df)
    poi_stats_df.to_csv(OUTPUT_DIR / 'poi_stats.csv')
