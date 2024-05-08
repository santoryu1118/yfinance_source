import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam

# Constants
TRAIN_TEST_SPLIT_RATIO = 0.75
EPOCHS = 5
BATCH_SIZE = 32
TIME_STEP = 10
MOVING_WINDOWS = [10, 20, 50]

script_dir = os.path.dirname(os.path.abspath(__file__))


class CustomMinMaxScaler:
    def __init__(self, feature_range, data_range):
        self.data_min_, self.data_max_ = data_range
        self.scale_ = (feature_range[1] - feature_range[0]) / (self.data_max_ - self.data_min_)
        self.min_ = feature_range[0] - self.data_min_ * self.scale_

    def transform(self, X):
        return self.min_ + X * self.scale_

    def inverse_transform(self, X_scaled):
        return (X_scaled - self.min_) / self.scale_


def add_moving_averages(dataframe, window_sizes=None):
    """Add moving average columns to the dataframe for each specified window size."""
    enhanced_df = dataframe.copy()
    if window_sizes:
        for window_size in window_sizes:
            column_name = f"moving_avg_{window_size}"
            enhanced_df[column_name] = enhanced_df['AdjClose'].rolling(window=window_size).mean()
    return enhanced_df


def shift_close_column(dataframe, shifts=1):
    """Shift the 'AdjClose' column in the dataframe by a specified number of shifts."""
    enhanced_df = dataframe.copy()
    for i in range(1, shifts + 1):
        enhanced_df[f'Close_t_{i}'] = enhanced_df['AdjClose'].shift(i)
    # Remove initial rows with NaN values resulting from the shift operation
    enhanced_df.dropna(inplace=True)
    return enhanced_df


def prepare_lstm_features(dataframe,
                          time_step=1,
                          window_sizes=None):
    """Prepares the dataset for LSTM training by adding moving averages and shifting the 'Close' column."""
    # Add moving averages
    enhanced_df = add_moving_averages(dataframe, window_sizes)

    # Shift the 'AdjClose' column and drop rows with NaN values
    shifted_df = shift_close_column(enhanced_df, time_step)

    # Ensure all necessary columns are included and return DataFrame ready for scaling
    feature_columns = [f'moving_avg_{ws}' for ws in window_sizes] + \
                      [f'Close_t_{i}' for i in range(1, time_step + 1)]
    features_df = shifted_df[feature_columns]

    # Extract labels before dropping 'AdjClose'
    labels = shifted_df['AdjClose'].values

    print(f"X.shape : {features_df.shape}, y.shape : {labels.shape}")

    return features_df, labels


def build_lstm_model(X_train, y_train, epochs, batch_size):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))

    # First LSTM layer with 50 units and returning sequences for the next LSTM layer
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    # Second LSTM layer with 20 units, not returning sequences as this is the last LSTM layer
    model.add(LSTM(units=20, return_sequences=False))
    # Dense layer to output the predicted value
    model.add(Dense(units=1))
    # Compile the model with Adam optimizer and MSE loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(f"model.summary : {model.summary()}")

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    # model.save_weights('my_model.h5')  # new_model.load_weights('iris_weight')
    model.save(os.path.join(script_dir, 'model.keras'))  # model = load_model('my_model.keras')

    return model


if __name__ == "__main__":
    train_data_csv = os.path.join(script_dir, '../postgres/data/train_yfdata.csv')
    df = pd.read_csv(train_data_csv, index_col="Datetime")

    # Split dataset into train and test
    train_data, test_data = np.split(df, [int(TRAIN_TEST_SPLIT_RATIO * len(df))])
    print(f"train_data.shape : {train_data.shape}, test_data.shape : {test_data.shape}")

    # LSTM are sensitive to the scale of the data. So normalize the data using Min-Max scaling
    # you must "learn" the scaling parameters (creating the scaler) only using your training dataset
    # apply the same scaler to the test dataset
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = CustomMinMaxScaler(feature_range=(0, 1), data_range=(140, 200))
    # train_data['AdjClose'] = scaler.fit_transform(train_data[['AdjClose']])
    train_data['AdjClose'] = scaler.transform(train_data[['AdjClose']])
    test_data['AdjClose'] = scaler.transform(test_data[['AdjClose']])
    # Save the scaler to a file
    joblib.dump(scaler, os.path.join(script_dir, 'scaler.save'))  # scaler = joblib.load('scaler.save')

    X_train, y_train = prepare_lstm_features(train_data, TIME_STEP, MOVING_WINDOWS)
    X_test, y_test = prepare_lstm_features(test_data, TIME_STEP, MOVING_WINDOWS)

    # Build and train LSTM model
    model = build_lstm_model(X_train, y_train, EPOCHS, BATCH_SIZE)

    # prediction and check performance metrics
    train_predict, test_predict = model.predict(X_train), model.predict(X_test)

    train_predict, test_predict = scaler.inverse_transform(train_predict), scaler.inverse_transform(test_predict)
    y_train, y_test = scaler.inverse_transform(y_train.reshape(-1, 1)), scaler.inverse_transform(y_test.reshape(-1, 1))

    final_df = pd.concat([
        pd.DataFrame(scaler.inverse_transform(X_test['Close_t_1'].values.reshape(-1, 1)), columns=['Close_t_1']),
        pd.DataFrame(y_test, columns=['Actual_Close']),
        pd.DataFrame(test_predict, columns=['Predicted_Close'])
    ], axis=1)
    # Add new columns as binary values (1 for True, 0 for False)
    final_df['Actual_Greater_than_Close_t_1'] = (final_df['Actual_Close'] > final_df['Close_t_1']).astype(int)
    final_df['Predicted_Greater_than_Close_t_1'] = (final_df['Predicted_Close'] > final_df['Close_t_1']).astype(int)
    final_df.to_csv(os.path.join(script_dir, 'test_predictions.csv'), index=False)

    actual_counts = final_df['Actual_Greater_than_Close_t_1'].value_counts()
    predicted_counts = final_df['Predicted_Greater_than_Close_t_1'].value_counts()
    print(actual_counts, predicted_counts)

    # Calculate RMSE performance metrics
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
    print(f"train_rmse : {train_rmse}, test_rmse : {test_rmse}")

    # trainPredictPlot = np.full_like(df, np.nan)
    # trainPredictPlot[TIME_STEP:len(train_predict) + TIME_STEP, :] = train_predict
    # testPredictPlot = np.full_like(df, np.nan)
    # testPredictPlot[-len(test_predict):, :] = test_predict
    # # plot baseline and predictions
    # plt.plot(df, label='Actual Price', color='blue')
    # plt.plot(trainPredictPlot, label='trainPredict Price', color='orange')
    # plt.plot(testPredictPlot, label='testPredict Price', color='green')
    # # plt.xlabel('Time')
    # # plt.ylabel('Close Price')
    # plt.show()
