from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.functions import vector_to_array

# https://spark.apache.org/docs/latest/ml-classification-regression.html

spark = SparkSession.builder \
    .appName("StockPrices") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.ui.port", "7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
# conf = spark.sparkContext.getConf()
# print(f"conf.getAll() : {conf.getAll()}")

schema = T.StructType([
    T.StructField("Datetime", T.TimestampType(), True),
    T.StructField("Open", T.DoubleType(), True),
    T.StructField("High", T.DoubleType(), True),
    T.StructField("Low", T.DoubleType(), True),
    T.StructField("Close", T.DoubleType(), True),
    T.StructField("Adj Close", T.DoubleType(), True),
    T.StructField("Volume", T.DoubleType(), True)])

yfinance_df = spark.read \
    .schema(schema) \
    .csv("./postgres/data/train_yfdata.csv", header="true")

# Define a window spec to order data
windowSpec = Window.orderBy("Datetime")

# 어제에 비해서 오늘 가격이 얼마나 올랐는지
# yfinance_df['percentChange'] = round(yfinance_df['Adj Close'] / yfinance_df['Adj Close'].shift(1) - 1, 4)
yfinance_df = yfinance_df.withColumn("percentChange",
                                     F.round((F.col("Adj Close") / F.lag("Adj Close").over(windowSpec)) - 1, 4))

# target == label : 내일 가격이 오늘에 비해서 + 인지 - 인지
# yfinance_df["Target"] = (yfinance_df['Adj Close'].shift(-1) > yfinance_df['Adj Close']).astype(int)
yfinance_df = yfinance_df.withColumn("Target",
                                     F.when(F.lead("Adj Close", 1).over(windowSpec) > F.col('Adj Close'), 1)
                                     .otherwise(0).cast(T.IntegerType()))

# Horizon for the rolling window
horizons = [2, 5, 10, 20]
moving_avg_predictors = []
trend_predictors = []
for horizon in horizons:
    # The range between - (horizon-1) and the current row (0) ensures a rolling window of the desired size
    moving_avg_window = Window.orderBy("Datetime").rowsBetween(-(horizon - 1), 0)
    moving_avg_column = f"Moving_Average_{horizon}"
    moving_avg_predictors.append(moving_avg_column)
    # Calculate the rolling average
    yfinance_df = yfinance_df.withColumn(moving_avg_column, F.avg("Adj Close").over(moving_avg_window))

    rolling_window = Window.orderBy("Datetime").rowsBetween(-(horizon - 1), -1)
    trend_column = f"Trend_{horizon}"
    trend_predictors.append(trend_column)
    # Count of days in the past x days that the stock price went up by calculating the rolling sum of 'Target' column
    yfinance_df = yfinance_df.withColumn(trend_column, F.sum("Target").over(rolling_window))

print(f"moving_avg_predictors : {moving_avg_predictors}")
print(f"trend_predictors : {trend_predictors}")


scale_columns = ["High", "Low", "Adj Close", "Volume"] + moving_avg_predictors
# Scaling은 일반 컬럼형(숫자형)이 아니라 vector형에만 적용이 가능함
vec_assembler = VectorAssembler(inputCols=scale_columns, outputCol='features')
# yfinance_df_vectorized = vec_assembler.transform(yfinance_df)

# vector화된 컬럼에 대해서 StandardScaler 적용
minmax_scaler = MinMaxScaler(inputCol='features', outputCol='minmax_scaled_features')
# standard_scaler_model = minmax_scaler.fit(yfinance_df_vectorized)
# standard_scaled_df = standard_scaler_model.transform(yfinance_df_vectorized)

pipeline = Pipeline(stages=[vec_assembler, minmax_scaler])
scaled_yfinance_df = pipeline.fit(yfinance_df).transform(yfinance_df)

scaled_yfinance_df = scaled_yfinance_df.withColumn("minmax_scaled_features", vector_to_array(F.col("minmax_scaled_features")))
scaled_predictors = []
for i, col_name in enumerate(scale_columns):
    scaled_column = f"{col_name}_scaled"
    scaled_predictors.append(scaled_column)
    scaled_yfinance_df = scaled_yfinance_df.withColumn(scaled_column, F.col("minmax_scaled_features")[i])
scaled_yfinance_df = scaled_yfinance_df.drop("features", "minmax_scaled_features")

print(f"Number of rows before dropping nulls: {scaled_yfinance_df.count()}")
scaled_yfinance_df = scaled_yfinance_df.dropna()
print(f"Number of rows after dropping nulls: {scaled_yfinance_df.count()}")
print('+++++++++++++++++++++++++++')

feature_columns = scaled_predictors + trend_predictors
print(f"feature_columns : {feature_columns}")
# feature_columns = ["High_scaled", "Low_scaled", "Adj Close_scaled", "Volume_scaled", "percentChange"]

"""
Model Training 시작
"""
vec_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html
# probability가 threshold 이상이어야 label 1을 받음
lr = LogisticRegression(featuresCol='features', labelCol='Target', threshold=0.55, maxIter=10)
pipeline = Pipeline(stages=[vec_assembler, lr])

# ML 알고리즘 객체의 fit() 메소드를 이용하여 train feature vector 데이터 세트를 학습하고 이를 ML Model로 반환함.
pipeline_model = pipeline.fit(scaled_yfinance_df)
# "rawPrediction", "probability", "prediction" column들이 생김
# probability: [0.3711488123012673,0.6288511876987327] / probability[0] 은 0번째 label의 확률, [1]은 1번째 label의 확률
predictions = pipeline_model.transform(scaled_yfinance_df)
print("predictions", predictions)
predictions.select("probability", "prediction", "target").show(truncate=False)


predictions.groupBy("Target").count().show()
predictions.groupBy("prediction").count().show()

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='Target', predictionCol='prediction',
                                                       metricName='accuracy')
evaluator_precision = MulticlassClassificationEvaluator(labelCol='Target', predictionCol='prediction',
                                                        metricName='weightedPrecision')
evaluator_recall = MulticlassClassificationEvaluator(labelCol='Target', predictionCol='prediction',
                                                     metricName='weightedRecall')
evaluator_f1_score = MulticlassClassificationEvaluator(labelCol='Target', predictionCol='prediction', metricName='f1')
print('정확도:', evaluator_accuracy.evaluate(predictions))
print('정밀도(precision):', evaluator_precision.evaluate(predictions))
print('재현율(recall):', evaluator_recall.evaluate(predictions))
print('f1 score:', evaluator_f1_score.evaluate(predictions))

# Access the trained LogisticRegressionModel from the PipelineModel
lr_model = pipeline_model.stages[-1]
# Print the coefficients of the Logistic Regression model along with the feature names
for name, value in zip(feature_columns, lr_model.coefficients):
    print(f'name: {name}, value: {value}')

