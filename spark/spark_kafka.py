# spark-submit --master spark://spark-master:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 work/spark_kafka.py
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("yfinance_spark") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .config("spark.sql.shuffle.partitions", "3") \
    .getOrCreate()
    # .config("spark.jars.package", 'org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1') \

events = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker:29092") \
    .option("subscribe", "test-target-topic") \
    .option("startingOffsets", "earliest") \
    .load()


def process_batch(df, epoch):
    # Perform any transformations or actions you want here
    # For example, show the DataFrame
    df.show()


# Write the output to console sink to check the output
writing_df = events.writeStream \
    .format("console") \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()
# .option("checkpointLocation", "checkpoint_dir") \

print('cccccccccccc')

# Start the streaming application to run until the following happens
# 1. Exception in the running program
# 2. Manual Interruption
writing_df.awaitTermination()
