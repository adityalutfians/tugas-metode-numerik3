import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Membaca dataset dari file CSV lokal
data = pd.read_csv("student_performance.csv")

# Memilih kolom yang relevan
NL = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
NT = data['Performance Index'].values

# Model Linear (Metode 1)
linear_model = LinearRegression()
linear_model.fit(NL, NT)
NT_pred_linear = linear_model.predict(NL)

# Menghitung RMS untuk Model Linear
rms_linear = mean_squared_error(NT, NT_pred_linear, squared=False)

# Plot hasil regresi linear
plt.scatter(NL, NT, color='blue', label='Data asli')
plt.plot(NL, NT_pred_linear, color='red', label='Regresi Linear')
plt.title('Regresi Linear: Jumlah Latihan Soal vs Nilai Ujian')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.show()

# Model Eksponensial (Metode 3)
def exponential_model(x, a, b):
    return a * np.exp(b * x)

params, covariance = curve_fit(exponential_model, NL.flatten(), NT)
NT_pred_exponential = exponential_model(NL, *params)

# Menghitung RMS untuk Model Eksponensial
rms_exponential = mean_squared_error(NT, NT_pred_exponential, squared=False)

# Plot hasil regresi eksponensial
plt.scatter(NL, NT, color='blue', label='Data asli')
plt.plot(NL, NT_pred_exponential, color='green', label='Regresi Eksponensial')
plt.title('Regresi Eksponensial: Jumlah Latihan Soal vs Nilai Ujian')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.show()

# Menampilkan hasil RMS
print(f'RMS Model Linear: {rms_linear}')
print(f'RMS Model Eksponensial: {rms_exponential}')