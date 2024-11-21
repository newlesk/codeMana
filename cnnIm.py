import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูลจำลองสำหรับ Feature Map (Input)
feature_map = np.random.rand(10, 8)  # 10 Timesteps x 8 Features

# MaxPooling จำลอง: ลด Timesteps ลงครึ่งหนึ่ง
pooled_map = feature_map[::2]  # เลือกทุกๆ 2 แถว (Window Size=2)

# Dropout จำลอง: ปิดการใช้งานบาง Features แบบสุ่ม
dropout_rate = 0.5
dropout_map = pooled_map * (np.random.rand(*pooled_map.shape) > dropout_rate)

# สร้างภาพแสดงกระบวนการ
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ภาพแรก: Feature Map (Input)
axes[0].imshow(feature_map, aspect='auto', cmap='Blues')
axes[0].set_title("Input Feature Map")
axes[0].set_xlabel("Features")
axes[0].set_ylabel("Timesteps")

# ภาพที่สอง: MaxPooling
axes[1].imshow(pooled_map, aspect='auto', cmap='Greens')
axes[1].set_title("After MaxPooling")
axes[1].set_xlabel("Features")
axes[1].set_ylabel("Timesteps")

# ภาพที่สาม: Dropout
axes[2].imshow(dropout_map, aspect='auto', cmap='Oranges')
axes[2].set_title("After Dropout")
axes[2].set_xlabel("Features")
axes[2].set_ylabel("Timesteps")

plt.tight_layout()
plt.show()
