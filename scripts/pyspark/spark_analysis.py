import __main__

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf, row_number, count, min, stddev, avg, max, col, round
from pyspark.sql.window import Window

from math import radians, cos, sin, asin, sqrt, pi


def _start_spark(app_name='my_spark_app', master='local[*]'):
    """

    :param app_name:
    :param master:
    :return:
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

    :param data_path:
    :param args:
    :return:
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

    :param df:
    :param df_name:
    :return:
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


def _associate_poiid_to_request(poi_df, data_df):
    """

    :param poi_df:
    :param data_df:
    :return:
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

    :param df:
    :return:
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

    :return:
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
    poi_stats_df \
        .coalesce(1) \
        .write \
        .option('header', 'true') \
        .csv(OUTPUT_DIR / 'output.csv')

    spark.stop()
    return None


if __name__ == "__main__":
    main()
