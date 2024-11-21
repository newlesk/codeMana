import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

# สร้างข้อมูลจำลอง (Mock Data) สำหรับ Key Points
# ข้อมูล: 10 ตัวอย่าง, 10 Key Points, แต่ละจุดมี 2 มิติ (X, Y)
np.random.seed(0)
input_data = np.random.rand(10, 10, 2)  # (samples, timesteps, features)

# สร้างโมเดล CNN ขนาดเล็ก
model = Sequential()

# Layer 1: Conv1D
model.add(Conv1D(filters=4, kernel_size=2, activation='relu', input_shape=(10, 2)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Layer 2: Conv1D + MaxPooling
model.add(Conv1D(filters=8, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Layer 3: Flatten + Fully Connected
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Output 3 Classes

# แสดงสรุปโมเดล
model.summary()

# จำลองผล Feature Map หลังจาก Conv1D แต่ละ Layer
feature_map_1 = np.random.rand(5, 4)  # Feature Map หลัง Layer 1 (4 Filters, ลดขนาด Timesteps)
feature_map_2 = np.random.rand(2, 8)  # Feature Map หลัง Layer 2 (8 Filters, ลดขนาด Timesteps)

# สร้างภาพ CNN Block
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# แสดง Feature Map หลัง Conv1D Layer 1
axes[0].imshow(feature_map_1.T, aspect='auto', cmap='viridis')
axes[0].set_title('Feature Map after Conv1D Layer 1')
axes[0].set_xlabel('Timesteps')
axes[0].set_ylabel('Filters (4)')

# แสดง Feature Map หลัง Conv1D Layer 2
axes[1].imshow(feature_map_2.T, aspect='auto', cmap='viridis')
axes[1].set_title('Feature Map after Conv1D Layer 2')
axes[1].set_xlabel('Timesteps')
axes[1].set_ylabel('Filters (8)')

plt.tight_layout()
plt.show()
