import numpy as np
import matplotlib.pyplot as plt

# ตัวอย่างข้อมูล Input Sequence
timesteps = 30
features = 63
sequence_data = np.random.rand(timesteps, features)

# LSTM Output (จำลอง Temporal Representation)
lstm_units = 32
lstm_output = np.random.rand(lstm_units)

# สร้างภาพ
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Step 1: Input Sequence
axes[0, 0].imshow(sequence_data.T, aspect='auto', cmap='Blues')
axes[0, 0].set_title('Step 1: Input Sequence')
axes[0, 0].set_xlabel('Timesteps')
axes[0, 0].set_ylabel('Features')

# Step 2: LSTM Process
axes[0, 1].plot(range(1, timesteps + 1), np.mean(sequence_data, axis=1), label='Mean Feature Value', color='green')
axes[0, 1].set_title('Step 2: LSTM Process (Temporal Dependency)')
axes[0, 1].set_xlabel('Timesteps')
axes[0, 1].set_ylabel('Aggregated Features')
axes[0, 1].legend()

# Step 3: Temporal Representation
axes[1, 0].bar(range(lstm_units), lstm_output, color='orange')
axes[1, 0].set_title('Step 3: LSTM Output (Temporal Representation)')
axes[1, 0].set_xlabel('LSTM Units')
axes[1, 0].set_ylabel('Values')

# Step 4: Fully Connected Layer Input
axes[1, 1].scatter(range(lstm_units), lstm_output, color='purple')
axes[1, 1].set_title('Step 4: Fully Connected Layer Input')
axes[1, 1].set_xlabel('LSTM Units')
axes[1, 1].set_ylabel('Values')

plt.tight_layout()
plt.show()
