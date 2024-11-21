import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# ใช้ Mediapipe สำหรับ Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ตัวแสดงภาพสำหรับ Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# โมเดล LSTM สำหรับ Temporal Module
model = Sequential()
model.add(LSTM(16, input_shape=(30, 63), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(3, activation='softmax'))

# เก็บข้อมูล Key Points (Time Series)
sequence_data = []
max_frames = 30  # จำนวนเฟรมที่เก็บไว้

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(1)

def process_frame(image, results):
    """ดึงข้อมูล Key Points และแสดงการเปลี่ยนแปลงใน Matplotlib"""
    global sequence_data
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # ดึงข้อมูล 21 Key Points (X, Y, Z)
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            # เพิ่มข้อมูลเข้า Sequence
            sequence_data.append(keypoints)
            if len(sequence_data) > max_frames:
                sequence_data.pop(0)
            
            # การ Normalize
            normalized_sequence = np.array(sequence_data) / np.max(sequence_data, axis=0)

            # Output จาก LSTM (Temporal Representation)
            if len(sequence_data) == max_frames:
                temporal_output = model.predict(np.expand_dims(normalized_sequence, axis=0))[0]
            else:
                temporal_output = np.zeros(3)  # กรณียังไม่มีข้อมูลเพียงพอ

            # อัปเดตภาพใน Matplotlib
            axes[0].cla()
            axes[1].cla()
            axes[2].cla()

            # Input Data (Key Points Time Series)
            axes[0].imshow(normalized_sequence.T, aspect='auto', cmap='Blues')
            axes[0].set_title('Input: Normalized Key Points')
            axes[0].set_xlabel('Timesteps')
            axes[0].set_ylabel('Features')

            # Temporal Representation (LSTM)
            axes[1].bar(range(len(temporal_output)), temporal_output, color='green')
            axes[1].set_title('Temporal Module Output')
            axes[1].set_xlabel('Classes')
            axes[1].set_ylabel('Probability')

            # Real-Time Feedback
            axes[2].imshow(image[..., ::-1])  # BGR -> RGB
            axes[2].set_title('Real-Time Hand Tracking')
            axes[2].axis('off')

            plt.pause(0.01)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็น RGB และส่งเข้า Mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # ประมวลผลและแสดงภาพ
    process_frame(image, results)

    # แสดงภาพจาก OpenCV
    cv2.imshow('Mediapipe Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # กด 'Esc' เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
plt.close()
