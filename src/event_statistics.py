import pyspark
import time
import pip
import pandas as pd
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import datediff, to_date, max as max_, lit, col, collect_list, row_number, concat_ws, format_number, concat, monotonically_increasing_id
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

   
def read_csv(spark, file, schema):
    return spark.read.csv(file, header=False, schema=schema)

def split_alive_dead(events, mortality):
    '''
    param: spart dataframe events: [petientid, eventid, etimestamp, value] and dataframe mortality: [patientid, mtimestamp, label]
    return: spark dataframe alive_evnets and dead_events

    Task1: This function needs to be completed.
    Split the events to two spark dataframes. One is for alive patients, and one is 
    for dead patients.
    Variables returned from this function are passed as input DataFrame for later.

   events = read_csv(spark, path1, schema1)
    events = events.select(events.patientid, events.eventid, to_date(events.timestamp).alias("etimestamp"), events.value)

    mortality = read_csv(spark, path2, schema2)
    mortality = mortality.select(mortality.patientid, to_date(mortality.timestamp).alias("mtimestamp"), mortality.label)

    alive_events, dead_events = split_alive_dead(events, mortality)
    '''
    dead = mortality.select("patientid", "label")
    events = events.join(dead, events.patientid == dead.patientid, 'left').select(events.patientid, events.eventid, events.etimestamp, events.value, dead.label)


    alive_events = events.filter(events.label.isNull())
    dead_events = events.filter("label == 1")

    alive_events = alive_events.select("patientid", "eventid", "etimestamp","value")
    dead_events = dead_events.select("patientid", "eventid", "etimestamp","value")

    return alive_events, dead_events

def event_count_metrics(alive_events, dead_events):
    '''
    param: two spark DataFrame: alive_events, dead_events
    return: two spark DataFrame

    Task 2: Event count metrics
    Compute average, min and max of event counts 
    for alive and dead patients respectively  
    +------+------+------+                                                   
    |   avg|   min|  max |
    +------+------+------+
    |value1|value2|value3|
    +------+------+------+
    note: 
    1.please keep same column name as example showed before!
    2.return two DataFrame for alive and dead patients' events respectively.
    3.average computed with avg(), DO NOT round the results.
    '''

    alive  = alive_events.groupBy('patientid').count()
    aliveMinCount = alive.agg({'count':'min'}).collect()[0][0]
    aliveMaxCount = alive.agg({'count':'max'}).collect()[0][0]
    aliveAvgCount = alive.agg({'count':'avg'}).collect()[0][0]
    aliveData = [(aliveAvgCount, aliveMinCount, aliveMaxCount)]

    dead  = dead_events.groupBy('patientid').count()
    deadMinCount = dead.agg({'count':'min'}).collect()[0][0]
    deadMaxCount = dead.agg({'count':'max'}).collect()[0][0]
    deadAvgCount = dead.agg({'count':'avg'}).collect()[0][0]
    deadData = [(deadAvgCount, deadMinCount, deadMaxCount)]


    columns = ["avg", "min", "max"]
    alive_statistics = spark.createDataFrame(data=aliveData, schema=columns)
    dead_statistics = spark.createDataFrame(data=deadData, schema=columns)

    return alive_statistics, dead_statistics


def encounter_count_metrics(alive_events, dead_events):
    '''
    param: two spark DataFrame: alive_events, dead_events
    return: two spark DataFrame

    Task3: Compute average, median, min and max of encounter counts 
    for alive and dead patients respectively
    +------+--------+------+------+                                             
    |  avg | median | min  | max  |
    +------+--------+------+------+
    |value1| value2 |value3|value4|
    +------+--------+------+------+
    note: 
    1.keep alive dataframe and dead dateframe respectively, in this case, you will get 2 (1 for alive and 1 for dead) dataframe.
    2.please keep same column name as example showed before!
    3.average computed with mean(), DO NOT need to round the results.

    '''
    alive  = alive_events.groupBy('patientid').agg(countDistinct('etimestamp'))
    
    aliveMinCount = alive.agg({'count(etimestamp)':'min'}).collect()[0][0]
    aliveMaxCount = alive.agg({'count(etimestamp)':'max'}).collect()[0][0]
    aliveAvgCount = alive.agg({'count(etimestamp)':'mean'}).collect()[0][0]
    aliveMedianCount = alive.agg( percentile_approx('count(etimestamp)', 0.5, lit(1000000))).collect()[0][0]
   

    aliveData = [(aliveAvgCount, aliveMedianCount, aliveMinCount, aliveMaxCount)]

    dead  = dead_events.groupBy('patientid').agg(countDistinct('etimestamp'))
    deadeMinCount = dead.agg({'count(etimestamp)':'min'}).collect()[0][0]
    deadMaxCount = dead.agg({'count(etimestamp)':'max'}).collect()[0][0]
    deadAvgCount = dead.agg({'count(etimestamp)':'mean'}).collect()[0][0]
    deadMedianCount = dead.agg( percentile_approx('count(etimestamp)', 0.5, lit(1000000))).collect()[0][0]
   
   

    deadData = [(deadAvgCount, deadMedianCount, deadeMinCount, deadMaxCount)]

    columns = ["avg", "median", "min", "max"]

    alive_encounter_res = spark.createDataFrame(data=aliveData, schema=columns)
    dead_encounter_res= spark.createDataFrame(data=deadData, schema=columns)

    return alive_encounter_res, dead_encounter_res


def record_length_metrics(alive_events, dead_events):
    '''
    param: two spark DataFrame:alive_events, dead_events
    return: two spark DataFrame

    Task4: Record length metrics
    Compute average, median, min and max of record lengths
    for alive and dead patients respectively
    +------+--------+------+------+                                             
    |  avg | median | min  | max  |
    +------+--------+------+------+
    |value1| value2 |value3|value4|
    +------+--------+------+------+
    note: 
    1.keep alive dataframe and dead dateframe respectively, in this case, you will get 2 (1 for alive and 1 for dead) dataframe.
    2.please keep same column name as example showed before!
    3.average computed with mean(), DO NOT round the results.

    '''
   
    aliveMinMax = alive_events.groupBy('patientid').agg( max('etimestamp').alias('max'), min('etimestamp').alias('min'))
   
    aliveStay = aliveMinMax.withColumn('stay', datediff(aliveMinMax.max, aliveMinMax.min)).select('stay')

    aliveAvgStay = aliveStay.agg({'stay':'mean'}).collect()[0][0]
    aliveMedianStay = aliveStay.agg( percentile_approx("stay", 0.5, lit(1000000))).collect()[0][0]
    aliveMinStay = aliveStay.agg({'stay':'min'}).collect()[0][0]
    aliveMaxStay = aliveStay.agg({'stay':'max'}).collect()[0][0]
 


    aliveData = [(aliveAvgStay, aliveMedianStay, aliveMinStay, aliveMaxStay)]

    deadMinMax = dead_events.groupBy('patientid').agg( max('etimestamp').alias('max'), min('etimestamp').alias('min'))

    deadStay = deadMinMax.withColumn('stay', datediff(deadMinMax.max, deadMinMax.min)).select('stay')

    deadAvgStay = deadStay.agg({'stay':'mean'}).collect()[0][0]
    deadMedianStay = deadStay.agg( percentile_approx("stay", 0.5, lit(1000000))).collect()[0][0]
    deadMinStay = deadStay.agg({'stay':'min'}).collect()[0][0]
    deadMaxStay = deadStay.agg({'stay':'max'}).collect()[0][0]

    deadData = [(deadAvgStay, deadMedianStay, deadMinStay, deadMaxStay)]

    columns = ["avg", "median", "min", "max"]
    alive_recordlength_res = spark.createDataFrame(data=aliveData, schema=columns)
    dead_recordlength_res = spark.createDataFrame(data=deadData, schema=columns)

    return alive_recordlength_res, dead_recordlength_res


def top5(events):

    top5_event_count  = events.groupBy('eventid').count().sort(col('count').desc()).limit(5).collect()

    return top5_event_count



def Common(alive_events, dead_events):
    '''
    param: two spark DataFrame: alive_events, dead_events
    return: six spark DataFrame
    Task 5: Common diag/lab/med
    Compute the 5 most frequently occurring diag/lab/med
    for alive and dead patients respectively
    +------------+----------+                                                       
    |   eventid  |diag_count|
    +------------+----------+
    |  DIAG999999|      9999|
    |  DIAG999999|      9999|
    |  DIAG999999|      9999|
    |  DIAG999999|      9999|
    |  DIAG999999|      9999|
    +------------+----------+

    +------------+----------+                                                       
    |   eventid  | lab_count|
    +------------+----------+
    |  LAB999999 |      9999|
    |  LAB999999 |      9999|   
    |  LAB999999 |      9999|
    |  LAB999999 |      9999|
    +------------+----------+

    +------------+----------+                                                       
    |   eventid  | med_count|
    +------------+----------+
    |  DRUG999999|      9999|
    |  DRUG999999|      9999|
    |  DRUG999999|      9999|
    |  DRUG999999|      9999|
    |  DRUG999999|      9999|
    +------------+----------+
    note:
    1.keep alive dataframe and dead dateframe respectively, in this case, you will get 6 (3 for alive and 3 for dead) dataframe.
    2.please keep same column name as example showed before!
    '''


    alive_diag_data = alive_events.filter("eventid like '%DIAG%'")
    alive_lab_data = alive_events.filter("eventid like '%LAB%'")
    alive_med_data = alive_events.filter("eventid like '%DRUG%'")
    
    dead_diag_data = dead_events.filter("eventid like '%DIAG%'")
    dead_lab_data = dead_events.filter("eventid like '%LAB%'")
    dead_med_data = dead_events.filter("eventid like '%DRUG%'")
    
    alive_diag_list = top5(alive_diag_data)
    dead_diag_list = top5(dead_diag_data)

    columns = ["eventid", "diag_count"]
    alive_diag= spark.createDataFrame(data=alive_diag_list, schema=columns)
    dead_diag= spark.createDataFrame(data=dead_diag_list, schema=columns)

   
    # lab count

    alive_lab_list = top5(alive_lab_data)
    dead_lab_list = top5(dead_lab_data)

    columns = ["eventid", "drug_count"]
    alive_lab= spark.createDataFrame(data=alive_lab_list, schema=columns)
    dead_lab= spark.createDataFrame(data=dead_lab_list, schema=columns)

    # med count
    alive_med_list = top5(alive_med_data)
    dead_med_list = top5(dead_med_data)

    columns = ["eventid", "med_count"]
    alive_med= spark.createDataFrame(data=alive_med_list, schema=columns)
    dead_med= spark.createDataFrame(data=dead_med_list, schema=columns)

    return alive_diag, alive_lab, alive_med, dead_diag, dead_lab, dead_med


def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.

    path1 = './data/events.csv'
    schema1 = StructType([
        StructField("patientid", IntegerType(), True),
        StructField("eventid", StringType(), True),
        StructField("eventdesc", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("value", FloatType(), True)])

    path2 = './data/mortality.csv'
    schema2 = StructType([
        StructField("patientid", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("label", IntegerType(), True)])

    events = read_csv(spark, path1, schema1)
    events = events.select(events.patientid, events.eventid, to_date(events.timestamp).alias("etimestamp"), events.value)

    mortality = read_csv(spark, path2, schema2)
    mortality = mortality.select(mortality.patientid, to_date(mortality.timestamp).alias("mtimestamp"), mortality.label)

    alive_events, dead_events = split_alive_dead(events, mortality)
    
    #Compute the event count metrics
    start_time = time.time()
    alive_statistics, dead_statistics = event_count_metrics(alive_events, dead_events)
    
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    
    alive_statistics.show()
    dead_statistics.show()
    
    
    #Compute the encounter count metrics
    start_time = time.time()
    alive_encounter_res, dead_encounter_res = encounter_count_metrics(alive_events, dead_events)
    end_time = time.time()
    
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    #print(encounter_count)
    alive_encounter_res.show()
    dead_encounter_res.show()

    
    #Compute record length metrics
    start_time = time.time()
    alive_recordlength_res, dead_recordlength_res = record_length_metrics(alive_events, dead_events)
    end_time = time.time()
    
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    alive_recordlength_res.show()
    dead_recordlength_res.show()
    

    
    #Compute Common metrics
    start_time = time.time()
    alive_diag, alive_lab, alive_med, dead_diag, dead_lab, dead_med = Common(alive_events, dead_events)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    alive_diag.show()
    alive_lab.show()
    alive_med.show()
    dead_diag.show()
    dead_lab.show()
    dead_med.show()
    

  

    

if __name__ == "__main__":
    main()

