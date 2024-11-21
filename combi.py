import numpy as np
import matplotlib.pyplot as plt

# ตัวอย่างข้อมูล CNN Features (Spatial Features)
cnn_timesteps = 15
cnn_filters = 16
cnn_output = np.random.rand(cnn_timesteps, cnn_filters)

# ตัวอย่างข้อมูล LSTM Output (Temporal Representation)
lstm_units = 32
lstm_output = np.random.rand(lstm_units)

# ตัวอย่างผลลัพธ์การ Classification (Softmax Output)
classes = ["Fist", "Open Hand", "Point", "Victory"]
classification_output = np.array([0.1, 0.7, 0.15, 0.05])  # ตัวอย่าง Probabilities

# สร้างภาพ
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Step 1: CNN Features
axes[0, 0].imshow(cnn_output.T, aspect='auto', cmap='Greens')
axes[0, 0].set_title('Step 1: CNN Features (Spatial Features)')
axes[0, 0].set_xlabel('Timesteps')
axes[0, 0].set_ylabel('Filters')

# Step 2: LSTM Features
axes[0, 1].bar(range(lstm_units), lstm_output, color='orange')
axes[0, 1].set_title('Step 2: LSTM Output (Temporal Features)')
axes[0, 1].set_xlabel('LSTM Units')
axes[0, 1].set_ylabel('Values')

# Step 3: Combine CNN + LSTM
combined_features = np.concatenate([cnn_output.mean(axis=0), lstm_output])
axes[1, 0].bar(range(len(combined_features)), combined_features, color='purple')
axes[1, 0].set_title('Step 3: Combined Features (CNN + LSTM)')
axes[1, 0].set_xlabel('Combined Feature Units')
axes[1, 0].set_ylabel('Values')

# Step 4: Classification
axes[1, 1].bar(classes, classification_output, color='blue')
axes[1, 1].set_title('Step 4: Classification')
axes[1, 1].set_xlabel('Classes')
axes[1, 1].set_ylabel('Probability')

plt.tight_layout()
plt.show()
