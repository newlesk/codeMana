import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Flatten, Dense

# สร้างข้อมูลจำลองสำหรับ Key Points (Input)
# Input Data: 100 ตัวอย่าง, 21 Key Points, แต่ละจุดมี 3 มิติ (X, Y, Z)
np.random.seed(0)
input_data = np.random.rand(100, 21, 3)  # (samples, timesteps, features)

# สร้างโมเดล Model Training
model = Sequential()

# CNN Block
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(21, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Temporal Module (LSTM)
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))

# Fully Connected Layer
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Output 3 Classes

# แสดงสรุปโมเดล
model.summary()

# จำลอง Feature Map สำหรับแต่ละขั้นตอน
cnn_features = np.random.rand(10, 32)  # Output จาก CNN Block (10 Timesteps, 32 Filters)
lstm_features = np.random.rand(5, 32)  # Output จาก LSTM (5 Reduced Timesteps, 32 Features)

# สร้างภาพแสดง CNN Block และ Temporal Module
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ภาพ CNN Block Features
axes[0].imshow(cnn_features.T, aspect='auto', cmap='Blues')
axes[0].set_title('CNN Block: High-Level Spatial Features')
axes[0].set_xlabel('Timesteps (10)')
axes[0].set_ylabel('Filters (32)')

# ภาพ LSTM Temporal Representation
axes[1].imshow(lstm_features.T, aspect='auto', cmap='Greens')
axes[1].set_title('LSTM Temporal Representation')
axes[1].set_xlabel('Reduced Timesteps (5)')
axes[1].set_ylabel('Features (32)')

plt.tight_layout()
plt.show()
