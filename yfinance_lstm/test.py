# import os
# import sys
#
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#
# import json
# import math
# from confluent_kafka import Consumer, KafkaError, KafkaException
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# from keras.models import load_model
# import joblib
# from yfinance_lstm.model_pretrain import CustomMinMaxScaler
#
# script_dir = os.path.dirname(os.path.abspath(__file__))
#
# # Load the trained LSTM model and the scaler
# model = load_model(os.path.join(script_dir, 'model.keras'))
# scaler = joblib.load(os.path.join(script_dir, 'scaler.save'))
#
# # Kafka configuration settings
# config = {
#     'bootstrap.servers': 'localhost:9092',  # Kafka broker address
#     'group.id': 'mygroup11',  # Consumer group ID
#     'auto.offset.reset': 'earliest',
# }
#
# # Kafka consumer setup
# consumer = Consumer(config)
# consumer.subscribe(['success.yfinance'])
# TIME_STEP = 10
# BATCH_SIZE = 20
# batch_data = []
#
#
# def preprocess_data(new_data):
#     """ Prepare single message data for prediction. """
#     df = pd.DataFrame(new_data)
#     df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
#     df.set_index('datetime', inplace=True)
#     df.rename(columns={"adjclose": "AdjClose"}, inplace=True)
#
#     # unnest data from 'lastTenPrices' column
#     for i in range(TIME_STEP):
#         df[f'Close_t_{i + 1}'] = df['lastTenPrices'].apply(lambda x: x[i] if i < len(x) else None)
#
#     features = (['moving_avg_10', 'moving_avg_20', 'moving_avg_50'] +
#                 [f'Close_t_{i + 1}' for i in range(TIME_STEP)])
#
#     return df[features], df['AdjClose']
#
#
# def make_prediction(data):
#     """ Make a prediction based on preprocessed data. """
#     prediction = model.predict(data)
#
#     return scaler.inverse_transform(prediction)  # Inverse transform for scaled model output
#
#
# def train_model(model, X, y):
#     """Update the model with new data."""
#     model.fit(X, y, epochs=1, verbose=1)
#     return model
#
#
# def log_error_to_file(y_new, predicted_value):
#     print(f"predicted_value: {predicted_value}, y_new: {y_new.values}")
#     rmse = math.sqrt(mean_squared_error(y_new, predicted_value))
#     with open(os.path.join(script_dir, 'error_log.txt'), 'a') as file:
#         file.write(f"y_new: {y_new}, predict: {predicted_value}, rmse: {rmse}\n")
#
#
# def process_data(model, batch_data_list):
#     X_new, y_new = preprocess_data(batch_data_list)
#     predicted_value = make_prediction(X_new)
#     log_error_to_file(y_new, predicted_value)
#     batch_data_list = []  # Reset batch only after processing
#
#     # Retrain the model with new data
#     model = train_model(model, X_new, y_new)
#     print("Model retrained with new data")
#
#     return model, batch_data_list
#
#
# if __name__ == "__main__":
#
#     try:
#         while True:
#             msg = consumer.poll(timeout=1.0)
#             if msg is None:
#                 continue
#
#             if msg.error():
#                 if msg.error().code() == KafkaError._PARTITION_EOF:
#                     print('%% %s [%d] reached end at offset %d\n' %
#                           (msg.topic(), msg.partition(), msg.offset()))
#                     # Handle the end of partition event properly
#                     if len(batch_data) > 0:
#                         model, batch_data = process_data(model, batch_data)
#                 else:
#                     raise KafkaException(msg.error())
#             else:
#                 # Decode the message and add to batch
#                 new_data = json.loads(msg.value().decode('utf-8'))
#                 print(f"new_data: {new_data}")
#                 batch_data.append(new_data)
#                 if len(batch_data) >= BATCH_SIZE:
#                     model, batch_data = process_data(model, batch_data)
#
#     except KeyboardInterrupt:
#         print('Stopped by user')
#     finally:
#         # Clean up
#         consumer.close()
