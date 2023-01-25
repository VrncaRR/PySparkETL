from nose.tools import with_setup, eq_, ok_, nottest, assert_almost_equals, nottest,assert_is

from pyspark.sql.functions import datediff, to_date, max as max_, lit, col, collect_list, row_number, concat_ws, format_number, concat, monotonically_increasing_id
from src.etl import calculate_index_dates,filter_events, aggregate_events, generate_feature_mapping, normalization, svmlight_convert, svmlight_samples
import pyspark.sql.functions as F

# from src.etl import *
import datetime 
import pandas as pd

import os, errno
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Read CSV File into DataFrame').getOrCreate()

sc = spark.sparkContext

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

def read_csv(spark, file, schema):
    return spark.read.csv(file, header=False, schema=schema)

events = read_csv(spark, path1, schema1)
events = events.select(events.patientid, events.eventid, to_date(events.timestamp).alias("etimestamp"), events.value)

mortality = read_csv(spark, path2, schema2)
mortality = mortality.select(mortality.patientid, to_date(mortality.timestamp).alias("mtimestamp"), mortality.label)
    
def setup_module ():
    global deliverables_path 
    deliverables_path = 'tests/'

@nottest
def setup_index_date ():
    global events_df, mortality_df
    mortality_df = mortality
    events_df = events

@with_setup (setup_index_date)
def test_index_date ():
    # INPUT:

    # events_df read from events.csv
    # e.g.
    # +---------+------------+----------+-----+
    # |patientid|     eventid|etimestamp|value|
    # +---------+------------+----------+-----+
    # |    20459|DIAG42872402|1994-12-04|  1.0|

    # mortality_df read from mortality.csv
    # +---------+----------+-----+
    # |patientid|mtimestamp|label|
    # +---------+----------+-----+
    # |    13905|2000-01-30|    1|
    # |    18676|2000-02-03|    1|
    # |    20301|2002-08-08|    1|
    # +---------+----------+-----+

    # OUTPUT:
    # index_df
    # index_date is datetime.date format
    # e.g.
    # +---------+----------+
    # |patientid|index_date|
    # +---------+----------+
    # |    20459|2000-09-19|
    # |     5206|2000-08-04|
    # |    20301|2002-07-09|
    # |    13905|1999-12-31|
    # |    18676|2000-01-04|
    # +---------+----------+

    expected = [[20459, datetime.date(2000, 9, 19)], \
                [5206, datetime.date(2000, 8, 4)], \
                [20301, datetime.date(2002, 7, 9)], \
                [13905, datetime.date(1999, 12, 31)], \
                [18676, datetime.date(2000, 1, 4)]]
    # print(expected_indx_dates)

    indx_date_df = calculate_index_dates(events_df, mortality_df)
    # indx_date_df.show()

    temp = indx_date_df.select(["patientid","index_date"]).rdd.map(lambda line: [x for x in line]).collect()
    print(temp)
    res = True

    if len(expected) != len(temp):
        res = False

    for eve in temp:
        if eve not in expected:
            res = False
            break

    eq_(res, True, "Index dates do not match")

@nottest
def setup_filter_events():
    global events_df, index_df

    events_list = [((20459), 'DIAG42872402', '1994-12-04', 1.0),
                ((19992), 'DIAG42872403', '1995-12-04', 1.0),
                ((20459), 'DIAG42872404', '1999-12-04', 1.0),
                ((20459), 'DIAG42872405', '2000-12-04', 1.0)]
    columns_events = ["patientid", "eventid", "etimestamp", "value"]
    events_df = spark.createDataFrame(data=events_list, schema=columns_events)

    index_dates_list = [((20459),'2000-09-19'),
            ((19992), '1996-09-19')]

    columns_index = ["patientid", "index_date"]
    index_df = spark.createDataFrame(data=index_dates_list, schema=columns_index)
    
    
@with_setup (setup_filter_events)
def test_filter_events():
    # INPUT:
    # create events df
    # events
    # e.g.
    # +---------+------------+----------+-----+
    # |patientid|     eventid|etimestamp|value|
    # +---------+------------+----------+-----+
    # |    20459|DIAG42872402|1994-12-04|  1.0|
    # create index_date df
    # index_dates
    # +---------+----------+
    # |patientid|index_date|
    # +---------+----------+
    # |    20459|2000-09-19|

    # OUTPUT:
    # filtered
    # e.g.
    # +---------+--------------+----------+-----+----------+---------------+
    # |patientid|   eventid    |etimestamp|value|index_date|time_difference|
    # +---------+--------------+----------+-----+----------+---------------+
    # |    20459|'DIAG42872404'|1999-12-04|  1.0|2000-09-19|              0|
    # |    19992|'DIAG42872403'|1995-12-04|  1.0|1996-09-19|              0|
    # +---------+--------------+----------+-----+----------+---------------+

    expected = [[19992, 'DIAG42872403'], [20459, 'DIAG42872404']]

    filtered = filter_events(events_df, index_df)
    temp = filtered.select(["patientid","eventid"]).rdd.map(lambda line: [x for x in line]).collect()

    res = True

    if len(expected) != len(temp):
        res = False

    for eve in temp:
        if eve not in expected:
            res = False
            break

    eq_(res, True, "Events are not filtered correctly")

@nottest
def setup_aggregate_events():
    global filtered_df

    filtered = [(19992, 'DIAG42872403', '1995-12-04', 1.0, '2000-09-19', 1751),
        (19992, 'DIAG42872403', '1995-12-04', 1.0, '2000-09-19', 1751),
        (19992, 'DIAG42872403', '1995-12-04', 1.0, '2000-09-19', 1751),
        (19992, 'DIAG42872404', '1995-12-04', 1.0, '2000-09-19', 1751),
        (29993, 'DIAG42872403', '1995-12-04', 1.0, '2000-09-19', 1751),
        (29993, 'DIAG42872403', '1995-12-04', 1.0, '2000-09-19', 1751)]
    columns_filtered = ["patientid", "eventid", "etimestamp", "value", "index_date", "time_difference"]
    filtered_df = spark.createDataFrame(data=filtered, schema=columns_filtered)
    
# @nottest
# def teardown_filter_events():
    
@with_setup (setup_aggregate_events)
def test_aggregate_events():
    # INPUT:
    # filtered
    # e.g.
    # +---------+----------+----------+-----+----------+---------------+
    # |patientid|   eventid|etimestamp|value|index_date|time_difference|
    # +---------+----------+----------+-----+----------+---------------+
    # |    20459|LAB3013603|2000-09-19|  0.6|2000-09-19|              0|

    # OUTPUT:
    # patient_features
    # e.g.
    # +---------+------------+-------------+
    # |patientid|     eventid|feature_value|
    # +---------+------------+-------------+
    # |     5206|DRUG19065818|            1|
    # |     5206|  LAB3021119|            1|
    # |    20459|  LAB3013682|           11|    
    # +---------+------------+-------------+

    patient_features = aggregate_events(filtered_df)
    '''
    patient_features
    +---------+------------+-------------+                                          
    |patientid|     eventid|feature_value|
    +---------+------------+-------------+
    |    19992|DIAG42872403|            3|
    |    19992|DIAG42872404|            1|
    |    29993|DIAG42872403|            2|
    +---------+------------+-------------+
    '''
    temp = patient_features.select(["patientid","eventid", "feature_value"]).rdd.map(lambda line: [x for x in line]).collect()

    expected = [[19992, 'DIAG42872403', 3], [29993, 'DIAG42872403', 2], [19992, 'DIAG42872404', 1]]

    res = True
    if len(expected) != len(temp):
        res = False

    for feat in temp:
        if feat not in expected:
            res = False
            break

    print("Actual: ", end = " ")
    print(temp)

    print("Expected: ", end = " ")
    print(expected)

    eq_(res, True, "Features are not created correctly")


@nottest
def setup_feature_mapping():
    global normalized_df 

    normalized = [(19992, 'DIAG42872403', 1.000),
        (19992, 'DIAG42872404', 1.000),
        (19993, 'DIAG42872403', 0.667),
        (19993, 'LAB1234', 0.667)]
    columns_normalized = ["patientid", "eventid", "normalized_feature_value"]
    normalized_df = spark.createDataFrame(data=normalized, schema=columns_normalized)
    

@with_setup (setup_feature_mapping)
def test_feature_mapping():

    # INPUT:
    # normalized
    # e.g.
    # +---------+------------+------------------------+
    # |patientid|     eventid|normalized_feature_value|
    # +---------+------------+------------------------+
    # |     5206|DRUG19065818|                   1.000|
    # |     5206|  LAB3021119|                   1.000|
    # |    20459|  LAB3013682|                   0.379|   
    # +---------+------------+------------------------+

    # OUTPUT:
    # event_map
    # e.g.
    # +------------+---------+------------------------+-----------+
    # |     eventid|patientid|normalized_feature_value|event_index|
    # +------------+---------+------------------------+-----------+
    # |DRUG19065818|     5206|                   1.000|         81|
    # |  LAB3021119|     5206|                   1.000|        174|
    # |  LAB3013682|    20459|                   0.379|        157|   
    # +------------+---------+------------------------+-----------+


    mapping = generate_feature_mapping(normalized_df)

    '''
    +------------+---------+------------------------+-----------+                   
    |     eventid|patientid|normalized_feature_value|event_index|
    +------------+---------+------------------------+-----------+
    |DIAG42872403|    19992|                     1.0|          0|
    |DIAG42872404|    19992|                     1.0|          1|
    |DIAG42872403|    19993|                   0.667|          0|
    +------------+---------+------------------------+-----------+
    '''
    res = True
    temp = mapping.rdd.map(lambda x: (x['eventid'], int(x['event_index']))).collect()

    expected = [('DIAG42872403', 0), ('DIAG42872404', 1), ('LAB1234', 2)]

    res = True
    if len(expected) != len(temp):
        res = False

    for feat in temp:
        if feat not in expected:
            res = False
            break

    print("Actual: ", end = " ")
    print(temp)

    print("Expected: ", end = " ")
    print(expected)

    eq_(res, True, "Feature mapping is not correct!")
    
@nottest
def setup_normalization():
    global features_df 

    features = [(19992, 'DIAG42872403', 1),
        (19992, 'DIAG42872404', 999),
        (19993, 'DIAG42872403', 4)]
    columns_features = ["patientid", "eventid", "feature_value"]
    features_df = spark.createDataFrame(data=features, schema=columns_features)

@with_setup (setup_normalization)
def test_normalization():
    # INPUT:
    # patient_features
    # e.g.
    # +---------+------------+-------------+
    # |patientid|     eventid|feature_value|
    # +---------+------------+-------------+
    # |     5206|DRUG19065818|            1|
    # |     5206|  LAB3021119|            1|
    # |    20459|  LAB3013682|           11|   


    # OUTPUT:
    # normalized
    # e.g.
    # +---------+------------+------------------------+
    # |patientid|     eventid|normalized_feature_value|
    # +---------+------------+------------------------+
    # |     5206|DRUG19065818|                   1.000|
    # |     5206|  LAB3021119|                   1.000|
    # |    20459|  LAB3013682|                   0.379|   
    # +---------+------------+------------------------+

    normalized = normalization(features_df)
    normalized.show()
    '''
    +---------+------------+------------------------+                               
    |patientid|     eventid|normalized_feature_value|
    +---------+------------+------------------------+
    |    19992|DIAG42872403|                   0.250|
    |    19992|DIAG42872404|                   1.000|
    |    19993|DIAG42872403|                   1.000|
    +---------+------------+------------------------+    
    '''
    temp = normalized.select(["patientid","eventid","normalized_feature_value"]).rdd.map(lambda x: [int(x[0]), x[1], x[2]]).collect()
    print(temp)
    expected = [[19992, 'DIAG42872403', 0.250], [19993, 'DIAG42872403', 1.000], [19992, 'DIAG42872404', 1.000]]

    res = True
    if len(expected) != len(temp):
        res = False
        eq_(res, True, "Normalization is not correct!")

    expected_dict = {}
    for eve in expected:
        expected_dict[(eve[0], eve[1])] = eve[2]

    # print(expected_dict)

    for feat in temp:
        k = (feat[0], feat[1])
        if k not in expected_dict.keys():
            res = False
            eq_(res, True, "No such patient and event: " + str(feat[0]) + ", " + feat[1])
            break
        else:
            assert_almost_equals(expected_dict[k], feat[2], places=2, msg="UNEQUAL in normalized val, Expected:%s, Actual:%s" %(expected, temp))

@nottest
def setup_svm_convert():
    global normalized_df, mapping_df 

    normalized = [(19992, 'DIAG4', 1.000),
        (19992, 'DIAG8', 1.000),
        (19993, 'LAB3', 0.500),
        (19993, 'DIAG4', 0.667)]
    columns_normalized = ["patientid", "eventid", "normalized_feature_value"]
    normalized_df = spark.createDataFrame(data=normalized, schema=columns_normalized)

    mapping = [('DIAG4', 2),
        ('DIAG8', 9),
        ('LAB3', 12)]
    columns_mapping = ["eventid", "event_index"]
    mapping_df = spark.createDataFrame(data=mapping, schema=columns_mapping)

@with_setup (setup_svm_convert)
def test_svm_convert():
    # INPUT:
    # normalized
    # e.g.
    # +---------+------------+------------------------+
    # |patientid|     eventid|normalized_feature_value|
    # +---------+------------+------------------------+
    # |    20459|  LAB3023103|                   0.062|
    # |    20459|  LAB3027114|                   1.000|
    # |    20459|  LAB3007461|                   0.115|
    # +---------+------------+------------------------+

    # event_map
    # e.g.
    # +----------+-----------+
    # |   eventid|event_index|
    # +----------+-----------+
    # |DIAG132797|          0|
    # |DIAG135214|          1|
    # |DIAG137829|          2|
    # +----------+-----------+

    # OUTPUT:    
    # svmlight: patientid, sparse_feature
    # sparse_feature is a list containing: features
    # earch feature is a string: "event_index:normalized_feature_val"
    # e.g
    # +---------+-------------------+
    # |patientid|   sparse_feature  |
    # +---------+-------------------+
    # |    19992|[2:1.000, 9:1.000] |
    # |    19993|[2:0.667, 12:0.500]|
    # +---------+-------------------+

    svmlight = svmlight_convert(normalized_df, mapping_df)
    svmlight.show()

 
    res = True
    temp = svmlight.rdd.map(lambda x: (x['patientid'], x['sparse_feature'])).collect()

    expected = {19992:["2:1.000", "9:1.000"], 19993:["2:0.667", "12:0.500"]}

    res = True
    if len(expected) != len(temp):
        res = False

    for pid, feat in temp:
        if pid not in expected.keys():
            res = False
            break
        elif feat != expected[pid]:
            res = False
            break

    print("Actual: ", end = " ")
    print(temp)
    expected_tuples = []
    for p in zip(expected.keys(), expected.values()):
        expected_tuples.append(p)
    # expected_tuples = [(19992, ["2:1.000", "9:1.000"]), (19993, ["2:0.667", "12:0.500"])]

    print("Expected: ", end = " ")
    print((expected_tuples))

    eq_(res, True, "Svmlight conversion is not correct!")


@nottest
def setup_svm_samples():
    global svmlight_df, mortality_df

    svmlight_data = [(19993, ["1:0.500", "97:0.667"]),
                  (19992, ["2:1.000", "2001:1.000"])]
    columns_svmlight = ["patientid", "sparse_feature"]
    svmlight_df = spark.createDataFrame(data=svmlight_data, schema=columns_svmlight)

    mortality_data = [(19993, datetime.date(2000, 9, 19), 1)]
    columns_mortality = ["patientid", "mtimestamp", "label"]
    mortality_df = spark.createDataFrame(data=mortality_data, schema=columns_mortality)

@with_setup(setup_svm_samples)
def test_svm_samples():
    # INPUT:
    # svmlight
    # +---------+--------------------+
    # |patientid|      sparse_feature|
    # +---------+--------------------+
    # |     5206|[4:1.000, 5:1.000...|
    # |    13905|[1:1.000, 11:1.00...|
    # |    18676|[0:1.000, 2:1.000...|
    # |    20301|[10:1.000, 12:1.0...|
    # |    20459|[136:0.250, 137:1...|
    # +---------+--------------------+

    # mortality
    # +---------+----------+-----+
    # |patientid|mtimestamp|label|
    # +---------+----------+-----+
    # |    13905|2000-01-30|    1|
    # |    18676|2000-02-03|    1|
    # |    20301|2002-08-08|    1|
    # +---------+----------+-----+

    # Task: create a new DataFrame by adding a new colum in "svmlight".
    # New column name is "save_feature" which is a String including target 
    # and sparse feature in SVMLight format;
    # New DataFrame name is "samples"
    # You can have other columns in "samples"

    # OUTPUT
    # samples
    # +---------+--------------------+-------------+--------------------+
    # |patientid|      sparse_feature|other columns|        save_feature|
    # +---------+--------------------+-------------+--------------------+
    # |     5206|[4:1.000, 5:1.000...|     ...     |0 4:1.000 5:1.000...|
    # |    13905|[1:1.000, 11:1.00...|     ...     |1 1:1.000 11:1.00...|
    # |    18676|[0:1.000, 2:1.000...|     ...     |1 0:1.000 2:1.000...|
    # |    20301|[10:1.000, 12:1.0...|     ...     |1 10:1.000 12:1.0...|
    # |    20459|[136:0.250, 137:1...|     ...     |0 136:0.250 137:1...|
    # +---------+--------------------+-------------+--------------------+
    # Hint:
    #         pyspark.sql.functions: concat_with

    expected = [[19992, "0 2:1.000 2001:1.000"], [19993, "1 1:0.500 97:0.667"]]

    samples = svmlight_samples(svmlight_df, mortality_df)
    temp = samples.rdd.map(lambda x: [x['patientid'], x['save_feature']]).collect()
    print(temp)
    res = True
    if len(expected) != len(temp):
        res = False

    for feat in temp:
        if feat not in expected:
            res = False
            break

    print("Actual: ", end = " ")
    print(temp)

    print("Expected: ", end = " ")
    print(expected)

    eq_(res, True, "Svmlight feature string is not correct!")
