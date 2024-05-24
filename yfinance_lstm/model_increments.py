import os
import sys
import json
import math
from confluent_kafka import Consumer, KafkaError, KafkaException
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import joblib
from keras.optimizers import Adam

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from yfinance_lstm.model_pretrain import CustomMinMaxScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(script_dir, 'model.keras'))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

scaler = joblib.load(os.path.join(script_dir, 'scaler.save'))
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'mygroup9',
    'auto.offset.reset': 'earliest',
})
consumer.subscribe(['success.yfinance'])
BATCH_SIZE = 50
batch_data = []


def preprocess_data(data):
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.rename(columns={"adjclose": "AdjClose"}, inplace=True)

    df.sort_index(inplace=True)

    for i in range(10):  # Assuming TIME_STEP is 10
        df[f'Close_t_{i + 1}'] = df['lastTenPrices'].apply(lambda x: x[i] if i < len(x) else None)
    features = ['moving_avg_10', 'moving_avg_20', 'moving_avg_50'] + [f'Close_t_{i + 1}' for i in range(10)]

    df = df.dropna(subset=features)

    df[features] = scaler.transform(df[features])
    return df[features], df['AdjClose']


def make_prediction(data):
    predictions = model.predict(data)
    return scaler.inverse_transform(predictions)


def log_error_to_file(y_new, predicted_value):
    print(f"predicted_value: {predicted_value}, y_new: {y_new.values}")
    rmse = math.sqrt(mean_squared_error(y_new, predicted_value))

    data = pd.DataFrame({
        "datetime": y_new.index,
        "Actual": y_new.values,
        "Predicted": predicted_value.flatten()
    })

    rmse_row = pd.DataFrame({"Actual": "RMSE", "Predicted": rmse}, index=[""])
    data_with_rmse = pd.concat([data, rmse_row])

    output_path = os.path.join(script_dir, "streams_predictions.csv")
    data_with_rmse.to_csv(output_path, index=False, mode='a', header=False)


def handle_batch_data(batch_data):
    X_new, y_new = preprocess_data(batch_data)
    print(f"X_new.shape: {X_new.shape}")
    print(f"y_new.shape: {y_new.shape}")
    if X_new.shape[0] == 0:
        return []
    predicted_value = make_prediction(X_new)
    log_error_to_file(y_new, predicted_value)
    # model.fit(X_new, y_new, epochs=1, verbose=1)
    # model.train_on_batch(X_new, y_new)
    print("Model retrained with new data")
    return []


if __name__ == "__main__":
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"%% {msg.topic()} [{msg.partition()}] reached end at offset {msg.offset()}")
                    if batch_data:
                        batch_data = handle_batch_data(batch_data)
                else:
                    raise KafkaException(msg.error())
            else:
                new_data = json.loads(msg.value().decode('utf-8'))
                batch_data.append(new_data)
                if len(batch_data) >= BATCH_SIZE:
                    batch_data = handle_batch_data(batch_data)
    except KeyboardInterrupt:
        print('Stopped by user')
    finally:
        consumer.close()
