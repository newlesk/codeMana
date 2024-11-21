import numpy as np
import matplotlib.pyplot as plt

# ข้อมูลจำลอง: Sequence ของ CNN Features
# มี Sequence 10 เฟรม (Timesteps) และแต่ละเฟรมมี 8 Features (จาก CNN)
np.random.seed(0)
cnn_features = np.random.rand(10, 8)  # (Timesteps, Features)

# ผลลัพธ์จำลองหลังผ่าน LSTM
# LSTM จะสร้าง Temporal Representation ที่ลดลำดับเวลา
lstm_output = np.random.rand(5, 16)  # (Reduced Timesteps, LSTM Features)

# ผลลัพธ์จำลองหลังผ่าน Timeformer
# Timeformer ใช้ Attention Mechanism เพื่อสร้าง Temporal Representation
timeformer_output = np.random.rand(5, 16)  # (Reduced Timesteps, Timeformer Features)

# สร้างภาพ Temporal Module
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Input: Sequence ของ CNN Features
axes[0].imshow(cnn_features.T, aspect='auto', cmap='Blues')
axes[0].set_title('Input: CNN Features')
axes[0].set_xlabel('Timesteps (10)')
axes[0].set_ylabel('Features (8)')

# LSTM Output: Temporal Representation
axes[1].imshow(lstm_output.T, aspect='auto', cmap='Greens')
axes[1].set_title('Output: LSTM Temporal Representation')
axes[1].set_xlabel('Reduced Timesteps (5)')
axes[1].set_ylabel('LSTM Features (16)')

# Timeformer Output: Temporal Representation
axes[2].imshow(timeformer_output.T, aspect='auto', cmap='Purples')
axes[2].set_title('Output: Timeformer Temporal Representation')
axes[2].set_xlabel('Reduced Timesteps (5)')
axes[2].set_ylabel('Timeformer Features (16)')

plt.tight_layout()
plt.show()
