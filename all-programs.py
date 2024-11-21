import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูลจำลองสำหรับแต่ละส่วน
# 1. Input Layer: 21 Key Points (X, Y, Z) + ความสูง/ความกว้าง
input_data = np.random.rand(65)  # 65 Features (21 Key Points * 3 + Height/Width)

# 2. CNN Block: High-Level Spatial Features
cnn_features = np.random.rand(10, 16)  # (Timesteps, Filters after Conv1D)

# 3. Temporal Module: LSTM/Timeformer
temporal_features = np.random.rand(5, 32)  # (Reduced Timesteps, Temporal Features)

# 4. Fully Connected Layers: Flatten Temporal Features
flattened_features = np.random.rand(128)  # 128 Flattened Features

# 5. Output Layer: Classification or Regression
classification_output = [0.1, 0.7, 0.2]  # Example Probabilities for 3 Classes
regression_output = 45.7  # Example Degree Prediction

# สร้างภาพสำหรับแต่ละส่วน
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Input Layer
axes[0, 0].bar(range(len(input_data)), input_data, color='blue')
axes[0, 0].set_title('Input Layer: 65 Features')
axes[0, 0].set_xlabel('Features')
axes[0, 0].set_ylabel('Values')

# CNN Block
axes[0, 1].imshow(cnn_features.T, aspect='auto', cmap='Blues')
axes[0, 1].set_title('CNN Block: High-Level Spatial Features')
axes[0, 1].set_xlabel('Timesteps')
axes[0, 1].set_ylabel('Filters (16)')

# Temporal Module
axes[0, 2].imshow(temporal_features.T, aspect='auto', cmap='Greens')
axes[0, 2].set_title('Temporal Module: Temporal Representation')
axes[0, 2].set_xlabel('Reduced Timesteps')
axes[0, 2].set_ylabel('Temporal Features (32)')

# Fully Connected Layers
axes[1, 0].bar(range(len(flattened_features)), flattened_features, color='orange')
axes[1, 0].set_title('Fully Connected Layers: Flattened Features')
axes[1, 0].set_xlabel('Features')
axes[1, 0].set_ylabel('Values')

# Output Layer - Classification
axes[1, 1].bar(['Class 1', 'Class 2', 'Class 3'], classification_output, color='purple')
axes[1, 1].set_title('Output Layer: Classification')
axes[1, 1].set_ylabel('Probability')

# Output Layer - Regression
axes[1, 2].bar(['Predicted Angle'], [regression_output], color='red')
axes[1, 2].set_title('Output Layer: Regression')
axes[1, 2].set_ylabel('Angle (Degrees)')

plt.tight_layout()
plt.show()
