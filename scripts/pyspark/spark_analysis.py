import __main__

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf, row_number, count, min, stddev, avg, max, col, round
from pyspark.sql.window import Window

from math import radians, cos, sin, asin, sqrt, pi


def _start_spark(app_name='my_spark_app', master='local[*]'):
    """
    Start a Spark session on the worker node and register the Spark application with the cluster. The app_name argument
    will apply only when this is called from a script sent to spark-submit.

    :param app_name: A string representing the name of the Pyspark app
    :param master: Cluster connection details (defaults to local[*])
    :return: A Spark session
    """
    flag_repl = not (hasattr(__main__, '__file__'))

    if not flag_repl:
        spark_builder = (
            SparkSession
                .builder
                .appName(app_name)
        )
    else:
        spark_builder = (
            SparkSession
                .builder
                .master(master)
                .appName(app_name)
        )

    spark_sess = spark_builder.getOrCreate()
    return spark_sess


def _read_data(data_path, *args):
    """
    Read the data from a specified input path

    :param data_path: A path to the csv file being loaded into a Spark dataframe
    :param args: Column names for the dataframe created from the csv file
    :return: A Spark dataframe
    """
    file_location = data_path
    file_type = 'csv'

    infer_schema = 'true'
    first_row_is_header = 'true'
    delimiter = ','

    df = spark.read.format(file_type) \
        .option('header', first_row_is_header) \
        .option('inferSchema', infer_schema) \
        .option('sep', delimiter) \
        .load(file_location) \
        .toDF(*args)

    return df


def _preprocess_data(df, df_name):
    """
    Remove duplicate entries from a specific dataframe

    This function is more of a factory method, in that the name parameter changes the way in which the data is
    preprocessed. In the case of request data, all duplicate entries (including the original entry) are removed, while
    only the duplicate entries are removed for the POI data.

    :param df: A Spark dataframe that needs to be preprocessed
    :param df_name: A string to specify the kind of dat
    :return: A Spark dataframe, post-processing.
    """
    if df_name == 'POI':
        df_filtered = df.dropDuplicates(['Latitude', 'Longitude'])
    elif df_name == 'Data':
        window_spec = Window.partitionBy('TimeSt', 'Latitude', 'Longitude')

        df_filtered = df \
            .select(
                'ID',
                'TimeSt',
                'Latitude',
                'Longitude',
                count('*').over(window_spec).alias('count')
        ) \
            .where('count = 1') \
            .drop('count')
    else:
        raise Exception('An invalid parameter was passed to preprocess_data.')

    return df_filtered


@udf("double")
def _haversine_dist(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points (say, A and B) from their geographical coordinates using the Haversine
    formula

    This method is a Spark UDF, as indicated by the decorator, that returns a double data type

    :param lat1: Latitude of point A
    :param lon1: Longitude of point A
    :param lat2: Latitude of point B
    :param lon2: Longitude of point B
    :return: A double data type representing the distance in kilometres
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    R = 6371  # Radius of earth in kilometers
    return c * R


def _associate_poiid_to_request(poi_df, data_df):
    """
    Implement the assignment of a request ID to its nearest POI

    :param poi_df: A Spark dataframe containing POI information
    :param data_df: A Spark dataframe containing request ID information
    :return: A Spark datframe assigning a POI to a request ID, together with the distance between the two
    """
    poi_data_join_df = data_df.crossJoin(
        poi_df \
            .withColumnRenamed('Latitude', 'POI_Latitude') \
            .withColumnRenamed('Longitude', 'POI_Longitude')
    )

    window_spec_2 = Window.partitionBy('ID')

    result_df = poi_data_join_df \
        .select(
        'ID',
        'POIID',
        _haversine_dist('Latitude', 'Longitude', 'POI_Latitude', 'POI_Longitude').alias('distance'),
        min(_haversine_dist('Latitude', 'Longitude', 'POI_Latitude', 'POI_Longitude')) \
            .over(window_spec_2) \
            .alias('min_distance')
    ) \
        .where('distance = min_distance') \
        .drop('distance')

    return result_df


def _calculate_poi_stats(df):
    """
    Calculate all the relevant statistics for a given position of interest (POI)

    :param poi_df: A Spark dataframe containing information mapping a request ID to a POI together with the distance
    between the ID and the POI
    :return: A Spark dataframe containing information about the mean and standard deviations of the distances of all
    various requests, the maximum distance (radius), request counts, and density associated to a POI
    """
    poi_stats_df = df \
        .groupBy('POIID') \
        .agg(
            count('*').alias('req_count'),
            avg('min_distance').alias('mean_distance'),
            stddev('min_distance').alias('stddev_distance'),
            max('min_distance').alias('radius')
    ) \
        .select(
            'POIID',
            'req_count',
            'mean_distance',
            'stddev_distance',
            'radius',
            round(col('req_count') / (pi * col('radius') ** 2), 6).alias('density')
    )

    return poi_stats_df


def main():
    """
    Implement all the operations associated to the project

    :return: None
    """
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = PROJECT_DIR / 'data' / 'DataSample.csv'
    POI_PATH = PROJECT_DIR / 'data' / 'POIList.csv'
    OUTPUT_DIR = PROJECT_DIR / 'output'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = _start_spark(app_name='eq_works_pyspark_app')

    data_df = _read_data(DATA_PATH, 'ID', 'TimeSt', 'Country', 'Province', 'City', 'Latitude', 'Longitude')
    poi_df = _read_data(POI_PATH, 'POIID', 'Latitude', 'Longitude')

    data_df_filtered = _preprocess_data(data_df, 'Data')
    poi_df_filtered = _preprocess_data(poi_df, 'POI')

    matched_data_poi_df = _associate_poiid_to_request(poi_df_filtered, data_df_filtered)

    poi_stats_df = _calculate_poi_stats(matched_data_poi_df)

    # Coalesce is used here to output a single .csv file, rather than a folder containing partitioned files.
    # If the file is too large to be loaded into memory, then the coalesce can be commented out.
    poi_stats_df \
        .coalesce(1) \
        .write \
        .option('header', 'true') \
        .csv(OUTPUT_DIR / 'output.csv')

    spark.stop()
    return None


if __name__ == "__main__":
    main()
