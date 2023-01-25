from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest,assert_is
import sys
sys.path.append("..")
from src.event_statistics import read_csv, split_alive_dead, event_count_metrics, encounter_count_metrics,record_length_metrics,Common
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
spark = SparkSession.builder.appName('Read CSV File into DataFrame').getOrCreate()
sc = spark.sparkContext


def setup_module ():
	filepath= './sample_test/'   # please change to correct path!
	global alive_events, dead_events

	# You may change the following path variable in coding but switch it back when submission.
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

	alive_events, dead_events = split_alive_dead(events, mortality)


def test_event_count():

	actual_value_1 = 426.5
	actual_value_2 = 299
	actual_value_3 = 554
	actual_value_4 = 636.66666666
	actual_value_5 = 543
	actual_value_6 = 812

	alive_statistics, dead_statistics = event_count_metrics(alive_events, dead_events)

	avg_alive = alive_statistics.select('avg').collect()[0][0]
	min_alive = alive_statistics.select('min').collect()[0][0]
	max_alive = alive_statistics.select('max').collect()[0][0]
	avg_dead = dead_statistics.select('avg').collect()[0][0]
	min_dead = dead_statistics.select('min').collect()[0][0]
	max_dead = dead_statistics.select('max').collect()[0][0]

	assert_almost_equals( avg_alive ,actual_value_1,places=2, msg="UNEQUAL in live_avg, Expected:%s, Actual:%s" %(actual_value_1, avg_alive))
	assert_almost_equals( min_alive ,actual_value_2,places=2, msg="UNEQUAL in live_min, Expected:%s, Actual:%s" %(actual_value_2, min_alive))
	assert_almost_equals( max_alive ,actual_value_3,places=2, msg="UNEQUAL in live_max, Expected:%s, Actual:%s" %(actual_value_3, max_alive))
	assert_almost_equals( avg_dead ,actual_value_4,places=2, msg="UNEQUAL in dead_avg, Expected:%s, Actual:%s" %(actual_value_4, avg_dead))
	assert_almost_equals( min_dead ,actual_value_5,places=2, msg="UNEQUAL in dead_min, Expected:%s, Actual:%s" %(actual_value_5, min_dead))
	assert_almost_equals( max_dead ,actual_value_6,places=2, msg="UNEQUAL in dead_max, Expected:%s, Actual:%s" %(actual_value_6, max_dead))


def test_encounter_count():
	expected_value_1 = 19.0
	expected_value_2 = 7
	expected_value_3 = 7
	expected_value_4 = 31
	expected_value_5 = 15.0
	expected_value_6 = 11
	expected_value_7 = 11
	expected_value_8 = 23

	alive_encounter_res, dead_encounter_res = encounter_count_metrics(alive_events, dead_events)
	
	avg_alive_encounter_count = alive_encounter_res.select('avg').collect()[0][0]
	median_alive_encounter_count = alive_encounter_res.select('median').collect()[0][0]
	min_alive_encounter_count = alive_encounter_res.select('min').collect()[0][0]
	max_alive_encounter_count = alive_encounter_res.select('max').collect()[0][0]
	avg_dead_encounter_count = dead_encounter_res.select('avg').collect()[0][0]
	median_dead_encounter_count = dead_encounter_res.select('median').collect()[0][0]
	min_dead_encounter_count = dead_encounter_res.select('min').collect()[0][0]
	max_dead_encounter_count = dead_encounter_res.select('max').collect()[0][0]

	assert_almost_equals(expected_value_1, avg_alive_encounter_count,places=2, msg="UNEQUAL in avg_alive_encounter_count, Expected:%s, Actual:%s" %(expected_value_1, avg_alive_encounter_count))
	assert_almost_equals(expected_value_2, median_alive_encounter_count,places=2, msg="UNEQUAL in median_alive_encounter_count,  Expected:%s, Actual:%s" %(expected_value_2, median_alive_encounter_count))
	assert_almost_equals(expected_value_3, min_alive_encounter_count,places=2, msg="UNEQUAL in min_alive_encounter_count, Expected:%s, Actual:%s" %(expected_value_3, min_alive_encounter_count))
	assert_almost_equals(expected_value_4, max_alive_encounter_count,places=2, msg="UNEQUAL in max_alive_encounter_count, Expected:%s, Actual:%s" %(expected_value_4, max_alive_encounter_count))
	assert_almost_equals(expected_value_5, avg_dead_encounter_count,places=2, msg="UNEQUAL in avg_dead_encounter_count, Expected:%s, Actual:%s" %(expected_value_5, avg_dead_encounter_count))
	assert_almost_equals(expected_value_6, median_dead_encounter_count,places=2, msg="UNEQUAL in median_dead_encounter_count, Expected:%s, Actual:%s" %(expected_value_6, median_dead_encounter_count))
	assert_almost_equals(expected_value_7, min_dead_encounter_count,places=2, msg="UNEQUAL in min_dead_encounter_count, Expected:%s, Actual:%s" %(expected_value_7, min_dead_encounter_count))
	assert_almost_equals(expected_value_8, max_dead_encounter_count,places=2, msg="UNEQUAL in max_dead_encounter_count, Expected:%s, Actual:%s" %(expected_value_8, max_dead_encounter_count))


def test_record_length():

	expected_value_1 = 1061.0
	expected_value_2 = 6
	expected_value_3 = 6
	expected_value_4 = 2116
	expected_value_5 = 25.66666666
	expected_value_6 = 21
	expected_value_7 = 10
	expected_value_8 = 46

	alive_recordlength_res, dead_recordlength_res = record_length_metrics(alive_events, dead_events)

	avg_alive_rec_len = alive_recordlength_res.select('avg').collect()[0][0]
	median_alive_rec_len = alive_recordlength_res.select('median').collect()[0][0]
	min_alive_rec_len = alive_recordlength_res.select('min').collect()[0][0]
	max_alive_rec_len = alive_recordlength_res.select('max').collect()[0][0]
	avg_dead_rec_len = dead_recordlength_res.select('avg').collect()[0][0]
	median_dead_rec_len = dead_recordlength_res.select('median').collect()[0][0]
	min_dead_rec_len = dead_recordlength_res.select('min').collect()[0][0]
	max_dead_rec_len = dead_recordlength_res.select('max').collect()[0][0]

	assert_almost_equals(expected_value_1, avg_alive_rec_len,places=2, msg="UNEQUAL in avg_alive_rec_len, Expected:%s, Actual:%s" %(expected_value_1, avg_alive_rec_len))
	assert_almost_equals(expected_value_2, median_alive_rec_len,places=2, msg="UNEQUAL in median_alive_rec_len,  Expected:%s, Actual:%s" %(expected_value_2, median_alive_rec_len))
	assert_almost_equals(expected_value_3, min_alive_rec_len,places=2, msg="UNEQUAL in min_alive_rec_len, Expected:%s, Actual:%s" %(expected_value_3, min_alive_rec_len))
	assert_almost_equals(expected_value_4, max_alive_rec_len,places=2, msg="UNEQUAL in max_alive_rec_len, Expected:%s, Actual:%s" %(expected_value_4, max_alive_rec_len))
	assert_almost_equals(expected_value_5, avg_dead_rec_len,places=2, msg="UNEQUAL in avg_dead_rec_len, Expected:%s, Actual:%s" %(expected_value_5, avg_dead_rec_len))
	assert_almost_equals(expected_value_6, median_dead_rec_len,places=2, msg="UNEQUAL in  median_dead_rec_len, Expected:%s, Actual:%s" %(expected_value_6, median_dead_rec_len))
	assert_almost_equals(expected_value_7, min_dead_rec_len,places=2, msg="UNEQUAL in min_dead_rec_len, Expected:%s, Actual:%s" %(expected_value_7, min_dead_rec_len))
	assert_almost_equals(expected_value_8, max_dead_rec_len,places=2, msg="UNEQUAL in max_alive_rec_len, Expected:%s, Actual:%s" %(expected_value_8, max_alive_rec_len))


def test_Common():
	
	expected_value_1 = 1
	expected_value_2 = 3013682
	expected_value_3 = 4
	expected_value_4 = 2
	expected_value_5 = 55
	expected_value_6 = 956874
	
	alive_diag, alive_lab, alive_med, dead_diag, dead_lab, dead_med = Common(alive_events, dead_events)

	alive_diag_count = alive_diag.collect()[1][1]
	alive_event_id = alive_lab.collect()[2][0][3:]
	alive_med_count = alive_med.collect()[3][1]
	dead_event_id = dead_diag.collect()[0][1]
	dead_lab_count = dead_lab.collect()[4][1]
	dead_med_count = dead_med.collect()[0][0][4:]

	assert_almost_equals(expected_value_1, alive_diag_count,places=2, msg="UNEQUAL in alive_diag_count, Expected:%s, Actual:%s" %(expected_value_1, alive_diag_count))
	assert_almost_equals(expected_value_2, int(alive_event_id),places=2, msg="UNEQUAL in alive_event_id, Expected:%s, Actual:%s" %(expected_value_2, alive_event_id))
	assert_almost_equals(expected_value_3, alive_med_count,places=2, msg="UNEQUAL in alive_med_count, Expected:%s, Actual:%s" %(expected_value_3, alive_med_count))
	assert_almost_equals(expected_value_4, dead_event_id,places=2, msg="UNEQUAL in dead_event_id_diag, Expected:%s, Actual:%s" %(expected_value_4, dead_event_id))
	assert_almost_equals(expected_value_5, dead_lab_count,places=2, msg="UNEQUAL in dead_lab_count_lab, Expected:%s, Actual:%s" %(expected_value_5, dead_lab_count))
	assert_almost_equals(expected_value_6, int(dead_med_count),places=2, msg="UNEQUAL in dead_med_count_med, Expected:%s, Actual:%s" %(expected_value_6, dead_med_count))
