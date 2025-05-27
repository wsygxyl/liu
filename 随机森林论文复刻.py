import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#设置随机种子以保证结果可重复
np.random.seed(42)

#生成时间序列数据
time_steps = np.arange(0,100,0.1)
sine_wave = np.sin(time_steps)
cosine_wave = np.cos(time_steps)
noisy_signal = np.random.normal(0,1,len(time_steps)) + 0.5 * np.sin(2 * time_steps)

#创建数据集
data = pd.DateFrame({
  'Time': time_steps,
  'Sine': sine_wave,
  'Cosine': cosine_wave,
  'Noisy': noisy_signal
})

#数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Sine','Cosine','Noisy']])
scaled_data = pd.DataFrame(scaled_data,columns=['sine','Cosine','Noisy'])
scaled_data['Time'] = data['Time']

#可视化标准化后的数据
plt.figure(figsize=(14,8))
plt.plot(scaled_data['Time'], scaled_data['Sine'], label='Normalized Sine Wave')
plt.plot(scaled_data['Time'], scaled_data['Cosine'], label='Normalized Cosine Wave')
plt.plot(scaled_data['Time'], scaled_data['Noisy'], label='Normalized Noisy Signal')
plt.title('Normalized Multivariate Time Series Data')
plt.xlabel('Time')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()
