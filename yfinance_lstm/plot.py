import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, "streams_predictions.csv")

# CSV 파일 읽기 (열 이름을 수동으로 지정)
df = pd.read_csv(csv_file_path, header=None, names=['datetime', 'Actual', 'Predicted'])

# RMSE 행 제거 (필요에 따라 보존할 수도 있음)
rmse_rows = df[df['Actual'] == 'RMSE']
df = df[df['Actual'] != 'RMSE']

# 'datetime' 열을 datetime 타입으로 변환
df['datetime'] = pd.to_datetime(df['datetime'])

# 'Actual'과 'Predicted' 열을 float 타입으로 변환
df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
df['Predicted'] = pd.to_numeric(df['Predicted'], errors='coerce')

# 실제값과 예측값을 플롯
plt.figure(figsize=(14, 7))
plt.plot(df['datetime'], df['Actual'], label='Actual')
plt.plot(df['datetime'], df['Predicted'], label='Predicted')

# RMSE 값을 플롯에 표시
for index, row in rmse_rows.iterrows():
    plt.annotate(f'RMSE: {row["Predicted"]:.2f}', (df['datetime'].iloc[-1], float(row["Predicted"])), 
                 textcoords="offset points", xytext=(0, 10), ha='center')

# 플롯 제목 및 레이블 설정
plt.title('Actual vs Predicted Values')
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# 플롯 표시
plt.show()
