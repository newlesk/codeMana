import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูลจำลองสำหรับแต่ละขั้นตอน
timesteps = 10  # จำนวนเฟรม (Timesteps)
keypoints = 21  # จำนวน Key Points
features = 3  # X, Y, Z
cnn_filters = 16
lstm_units = 32

# Input Data Preparation
input_data = np.random.rand(timesteps, keypoints * features)  # ข้อมูล Key Points (X, Y, Z)

# Feature Engineering
angles = np.random.rand(timesteps) * 180  # มุมองศาที่คำนวณได้
normalized_data = input_data / np.max(input_data, axis=0)  # Normalize

# Temporal Data Creation
temporal_data = np.stack([normalized_data for _ in range(30)], axis=0)  # 30 Frames

# CNN Block Output
cnn_output = np.random.rand(10, cnn_filters)  # Output จาก CNN Block

# Temporal Module Output
temporal_representation = np.random.rand(5, lstm_units)  # Output จาก LSTM

# Classification Output
classification_output = [0.1, 0.7, 0.2]  # Probabilities ของแต่ละ Class

# Regression Output
regression_output = 45.7  # มุมองศาที่คาดการณ์ได้

# สร้างกราฟในแต่ละขั้นตอน
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Step 1: Input Data Preparation
axes[0, 0].imshow(input_data.T, aspect='auto', cmap='Blues')
axes[0, 0].set_title('Step 1: Input Data Preparation')
axes[0, 0].set_xlabel('Timesteps')
axes[0, 0].set_ylabel('Key Points (X, Y, Z)')

# Step 2: Feature Engineering
axes[0, 1].plot(angles, color='red')
axes[0, 1].set_title('Step 2: Feature Engineering')
axes[0, 1].set_xlabel('Timesteps')
axes[0, 1].set_ylabel('Angle (Degrees)')

# Step 3: Temporal Data Creation
axes[0, 2].imshow(temporal_data[:, :, 0], aspect='auto', cmap='Greens')
axes[0, 2].set_title('Step 3: Temporal Data Creation')
axes[0, 2].set_xlabel('Timesteps')
axes[0, 2].set_ylabel('Frames')

# Step 4: CNN Block
axes[1, 0].imshow(cnn_output.T, aspect='auto', cmap='Purples')
axes[1, 0].set_title('Step 4: CNN Block Output')
axes[1, 0].set_xlabel('Timesteps')
axes[1, 0].set_ylabel('Filters')

# Step 5: Temporal Module
axes[1, 1].imshow(temporal_representation.T, aspect='auto', cmap='Oranges')
axes[1, 1].set_title('Step 5: Temporal Module')
axes[1, 1].set_xlabel('Reduced Timesteps')
axes[1, 1].set_ylabel('LSTM Units')

# Step 6: Output Layer
axes[1, 2].bar(['Class 1', 'Class 2', 'Class 3'], classification_output, color='cyan')
axes[1, 2].text(0.5, regression_output, f"Angle: {regression_output}°", color='red', ha='center', fontsize=12)
axes[1, 2].set_title('Step 6: Output Layer')
axes[1, 2].set_ylabel('Probability')

plt.tight_layout()
plt.show()
