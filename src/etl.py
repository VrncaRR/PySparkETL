from pyspark.sql.functions import datediff, to_date, max as max_, lit, col, collect_list, row_number, concat_ws, format_number, concat, monotonically_increasing_id
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
import shutil
import os
from pyspark.sql.window import Window
import operator
import pyspark.sql.functions as F

spark = SparkSession.builder.appName('Read CSV File into DataFrame').getOrCreate()
sc = spark.sparkContext

def read_csv(spark, file, schema):
    return spark.read.csv(file, header=False, schema=schema)

def calculate_index_dates(events, mortality):
    '''
    INPUT1:
    events_df read from events.csv
    e.g.
    +---------+------------+----------+-----+
    |patientid|     eventid|etimestamp|value|
    +---------+------------+----------+-----+
    |    20459|DIAG42872402|1994-12-04|  1.0|
    
    INPUT2:
    mortality_df read from mortality.csv
    +---------+----------+-----+
    |patientid|mtimestamp|label|
    +---------+----------+-----+
    |    13905|2000-01-30|    1|

    OUTPUT:
    index_df
    index_date is datetime.date format
    e.g.
    +---------+----------+
    |patientid|index_date|
    +---------+----------+
    |    20459|2000-09-19|
    |    13905|1999-12-31|
    +---------+----------+
    '''

    dead = mortality.select("patientid", "label")
    events = events.join(dead, events.patientid == dead.patientid, 'left').select(events.patientid, events.etimestamp, dead.label)

    #label alive people
    alive_events = events.filter(events.label.isNull())

    # alive people index date
    alive_index = alive_events.groupBy('patientid').agg( max_('etimestamp').alias('index_date'))

    #dead people index date
    dead_index = mortality.select(mortality.patientid, col('mtimestamp').alias("index_date"))

    dead_index = dead_index.withColumn('index_date', F.date_add(dead_index['index_date'], -30))

    df = alive_index.union(dead_index)

    # The output coloumns should have the name (patientid, index_date)
    # index_dates = [(20459, '2000-09-19'),
    #     (5206, '2000-08-04')]
    # columns = ["patientid", "index_date"]
    # df = spark.createDataFrame(data=index_dates, schema=columns)
    return df

def filter_events(events, index_dates):
    # TODO: filtered events should have the same input column of original events, select the corresponding columns and revise test as well
    '''
    INPUT:
    events: created events df, e.g.
    +---------+------------+----------+-----+
    |patientid|     eventid|etimestamp|value|
    +---------+------------+----------+-----+
    |    20459|DIAG42872402|1994-12-04|  1.0|
    +---------+------------+----------+-----+
    
    index_dates: created index_date df, e.g
    +---------+----------+
    |patientid|index_date|
    +---------+----------+
    |    20459|2000-09-19|
    +---------+----------+

    OUTPUT:
    filtered: e.g.
    +---------+--------------+----------+-----+
    |patientid|   eventid    |etimestamp|value|
    +---------+--------------+----------+-----+
    |    20459|'DIAG42872404'|1999-12-04|  1.0|
    |    19992|'DIAG42872403'|1995-12-04|  1.0|
    +---------+--------------+----------+-----+
    '''
    # Remove the events that are not in the observation window


    # merge the events df with index date df

    events = events.join(index_dates, events.patientid == index_dates.patientid, 'left').drop(index_dates.patientid)

    events = events.withColumn('start_date', F.date_add(events['index_date'], -2000))

    filtered = events.filter(events.etimestamp >= events.start_date)

    filtered = filtered.filter(events.etimestamp <= events.index_date)

    filtered = filtered.select("patientid", "eventid", "etimestamp", "value")

    # filtered = [(20459, 'DIAG42872404', '1999-12-04', 1.0)]
    # columns = ["patientid", "eventid", "etimestamp", "value"]
    # df = spark.createDataFrame(data=filtered, schema=columns)
    return filtered

def aggregate_events(filtered):
    '''
    INPUT:
    filtered
    e.g.
    +---------+----------+----------+-----+
    |patientid|   eventid|etimestamp|value|
    +---------+----------+----------+-----+
    |    20459|LAB3013603|2000-09-19|  0.6|

    OUTPUT:
    patient_features
    e.g.
    +---------+------------+-------------+
    |patientid|     eventid|feature_value|
    +---------+------------+-------------+
    |     5206|DRUG19065818|            1|
    |     5206|  LAB3021119|            1|
    |    20459|  LAB3013682|           11|    
    +---------+------------+-------------+
    '''
    # Output columns should be (patientid, eventid, feature_value)


    aggregated = filtered.groupBy('patientid', 'eventid').agg( F.count('eventid').alias('feature_value'))


    # features = [(20459, 'LAB3013682', 11)]
    # columns = ["patientid", "eventid", "feature_value"]
    # df = spark.createDataFrame(data=features, schema=columns)
    return aggregated

def generate_feature_mapping(agg_events):
    '''
    INPUT:
    agg_events
    e.g.
    +---------+------------+-------------+
    |patientid|     eventid|feature_value|
    +---------+------------+-------------+
    |     5206|DRUG19065818|            1|
    |     5206|  LAB3021119|            1|
    |    20459|  LAB3013682|           11|
    +---------+------------+-------------+

    OUTPUT:
    event_map
    e.g.
    +----------+-----------+
    |   eventid|event_index|
    +----------+-----------+
    |DIAG132797|          0|
    |DIAG135214|          1|
    |DIAG137829|          2|
    |DIAG141499|          3|
    |DIAG192767|          4|
    |DIAG193598|          5|
    +----------+-----------+
    '''
    # Hint: pyspark.sql.functions: monotonically_increasing_id
    # Output colomns should be (eventid, event_index)

    event_map = agg_events.select('eventid').distinct().sort(col('eventid').asc())

    event_map = event_map.withColumn('event_index', monotonically_increasing_id())

    # event_map = [("DIAG132797", 0)]
    # columns = ["eventid", "event_index"]
    # df = spark.createDataFrame(data=event_map, schema=columns)
    return event_map

def normalization(agg_events):
    '''
    INPUT:
    agg_events
    e.g.
    +---------+------------+-------------+
    |patientid|     eventid|feature_value|
    +---------+------------+-------------+
    |     5206|DRUG19065818|            1|
    |     5206|  LAB3021119|            1|
    |    20459|  LAB3013682|           11|   


    OUTPUT:
    normalized
    e.g.
    +---------+------------+------------------------+
    |patientid|     eventid|normalized_feature_value|
    +---------+------------+------------------------+
    |     5206|DRUG19065818|                   1.000|
    |     5206|  LAB3021119|                   1.000|
    |    20459|  LAB3013682|                   0.379|   
    +---------+------------+------------------------+
    '''
    # Output columns should be (patientid, eventid, normalized_feature_value)
    # Note: round the normalized_feature_value to 3 places after decimal: use round() in pyspark.sql.functions

    # calculate min,max value for every event
    minMaxTable = agg_events.groupBy('eventid').agg( max_('feature_value').alias('max'), F.min('feature_value').alias('min'))

    #  join the min max table with event table
    agg_events = agg_events.join(minMaxTable, minMaxTable.eventid == agg_events.eventid, 'left').drop(minMaxTable.eventid)

    # calculate normalized_feature_value
    agg_events = agg_events.withColumn('normalized_feature_value', 
            F.round((agg_events.feature_value/ agg_events.max)*1.000, 3))

    normalized = agg_events.select('patientid', 'eventid', 'normalized_feature_value')

    # event_map = [("5206", "DRUG19065818", 1.000)]
    # columns = ["patientid", "eventid", "normalized_feature_value"]
    # df = spark.createDataFrame(data=event_map, schema=columns)
    return normalized

def svmlight_convert(normalized, event_map):
    '''
    INPUT:
    normalized
    e.g.
    +---------+------------+------------------------+
    |patientid|     eventid|normalized_feature_value|
    +---------+------------+------------------------+
    |    20459|  LAB3023103|                   0.062|
    |    20459|  LAB3027114|                   1.000|
    |    20459|  LAB3007461|                   0.115|
    +---------+------------+------------------------+

    event_map
    e.g.
    +----------+-----------+
    |   eventid|event_index|
    +----------+-----------+
    |DIAG132797|          0|
    |DIAG135214|          1|
    |DIAG137829|          2|
    +----------+-----------+

    OUTPUT:    
    svmlight: patientid, sparse_feature
    sparse_feature is a list containing: feature pairs
    earch feature pair is a string: "event_index:normalized_feature_val"
    e.g
    +---------+-------------------+
    |patientid|   sparse_feature  |
    +---------+-------------------+
    |    19992|[2:1.000, 9:1.000] |
    |    19993|[2:0.667, 12:0.500]|
    +---------+-------------------+
    '''
    # Output columns should be (patientid, sparse_feature)
    # Note: for normalized_feature_val, when convert it to string, save 3 digits after decimal including "0": use format_number() in pyspark.sql.functions
    # Hint:
    #         pyspark.sql.functions: concat_with(), collect_list()
    #         pyspark.sql.window: Window.partitionBy(), Window.orderBy()

    # exclude 0 feature value
    normalized = normalized.filter(normalized.normalized_feature_value > 0)
    events = normalized.join(event_map, normalized.eventid == event_map.eventid, 'left').drop(normalized.eventid)

    order_window = Window.partitionBy('patientid').orderBy(col('event_index').asc())
    # events = events.withColumn('row_num', F.row_number()
    #             .over(Window.partitionBy('patientid').orderBy(col('event_index').asc())))
    
    events = events.withColumn('concate_feature', F.concat_ws(':',events.event_index, format_number(events.normalized_feature_value, 3)))

    svmlight = events.withColumn('sparse_feature', collect_list(events.concate_feature)
                .over(order_window))\
                    .groupBy('patientid')\
                    .agg(F.max('sparse_feature').alias('sparse_feature'))

    return svmlight


def svmlight_samples(svmlight, mortality):
    '''
    INPUT:
    svmlight
    +---------+--------------------+
    |patientid|      sparse_feature|
    +---------+--------------------+
    |     5206|[4:1.000, 5:1.000...|
    |    13905|[1:1.000, 11:1.00...|
    |    18676|[0:1.000, 2:1.000...|
    |    20301|[10:1.000, 12:1.0...|
    |    20459|[136:0.250, 137:1...|
    +---------+--------------------+

    mortality
    +---------+----------+-----+
    |patientid|mtimestamp|label|
    +---------+----------+-----+
    |    13905|2000-01-30|    1|
    |    18676|2000-02-03|    1|
    |    20301|2002-08-08|    1|
    +---------+----------+-----+

    OUTPUT
    samples
    +---------+--------------------+-------------+--------------------+
    |patientid|      sparse_feature|other columns|        save_feature|
    +---------+--------------------+-------------+--------------------+
    |     5206|[4:1.000, 5:1.000...|     ...     |0 4:1.000 5:1.000...|
    |    13905|[1:1.000, 11:1.00...|     ...     |1 1:1.000 11:1.00...|
    |    18676|[0:1.000, 2:1.000...|     ...     |1 0:1.000 2:1.000...|
    |    20301|[10:1.000, 12:1.0...|     ...     |1 10:1.000 12:1.0...|
    |    20459|[136:0.250, 137:1...|     ...     |0 136:0.250 137:1...|
    +---------+--------------------+-------------+--------------------+
    '''

    # Task: create a new DataFrame by adding a new colum in "svmlight".
    # New column name is "save_feature" which is a String including target 
    # and sparse feature in SVMLight format;
    # New DataFrame name is "samples"
    # You can have other columns in "samples"    
    # Hint:
    #         pyspark.sql.functions: concat_with

    samples = svmlight.join(mortality, svmlight.patientid == mortality.patientid, 'left').drop(mortality.patientid)

    samples = samples.withColumn('label', F.when(samples.label == 1, 1).otherwise(0))


    join_udf = F.udf(lambda x: " ".join(x))

    samples = samples.withColumn('save_feature',F.concat_ws(" ", 'label', join_udf('sparse_feature')) )
    
    samples = samples.select(svmlight.patientid, samples.save_feature)

    return samples

def train_test_split(samples, train_path, test_path):
    
    # DO NOT change content below
    samples = samples.randomSplit([0.2, 0.8], seed=48)

    testing = samples[0].select(samples[0].save_feature)
    training = samples[1].select(samples[1].save_feature)

    #save training and tesing data
    if os.path.exists(train_path):
        shutil.rmtree(train_path)

    training.write.option("escape","").option("quotes", "").option("delimiter"," ").text(train_path)

    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    testing.write.option("escape","").option("quotes", "").option("delimiter"," ").text(test_path)
    


def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''

    path1 = './sample_test/sample_events.csv'
    schema1 = StructType([
        StructField("patientid", IntegerType(), True),
        StructField("eventid", StringType(), True),
        StructField("eventdesc", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("value", FloatType(), True)])

    path2 = './sample_test/sample_mortality.csv'
    schema2 = StructType([
        StructField("patientid", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("label", IntegerType(), True)])

    events = read_csv(spark, path1, schema1)
    events = events.select(events.patientid, events.eventid, to_date(events.timestamp).alias("etimestamp"), events.value)


    mortality = read_csv(spark, path2, schema2)
    mortality = mortality.select(mortality.patientid, to_date(mortality.timestamp).alias("mtimestamp"), mortality.label)

    index_dates = calculate_index_dates(events, mortality)
    print('index_dates')
    #index_dates.show()


    filtered = filter_events(events, index_dates)
    print('filtered')
    #filtered.show()
    
    agg_events = aggregate_events(filtered)
    print('agg_events')
    #agg_events.show()

    event_map = generate_feature_mapping(agg_events)
    print('event_map')
    #event_map.show()
    

    normalized = normalization(agg_events)
    print('normalized')
    #normalized.show()

    svmlight = svmlight_convert(normalized, event_map)
    print('svmlight')
    svmlight.show()

    samples = svmlight_samples(svmlight, mortality)
    print('svmlight samples')
    samples.show()
'''
    train_test_split(samples, './deliverables/training', './deliverables/testing')

'''
if __name__ == "__main__":
    main()
