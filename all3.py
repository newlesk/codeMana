import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten

# ใช้ Mediapipe สำหรับ Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# สร้าง CNN + LSTM โมเดล
model = Sequential([
    Conv1D(16, kernel_size=3, activation='relu', input_shape=(30, 63)),
    MaxPooling1D(pool_size=2),
    LSTM(32, return_sequences=False),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# สร้างภาพสำหรับแสดงผลใน Matplotlib
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# เก็บข้อมูล Key Points (Time Series)
sequence_data = []
max_frames = 30
classes = ['Class 1', 'Class 2', 'Class 3']

def process_frame(image, results):
    """ประมวลผลเฟรม"""
    global sequence_data
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # ดึงข้อมูล 21 Key Points (X, Y, Z)
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            # เก็บข้อมูลใน Sequence
            sequence_data.append(keypoints)
            if len(sequence_data) > max_frames:
                sequence_data.pop(0)

            # ตรวจสอบว่าข้อมูลครบ 30 เฟรม
            if len(sequence_data) == max_frames:
                # Normalize ข้อมูล
                normalized_sequence = np.array(sequence_data) / np.max(sequence_data, axis=0)
                normalized_sequence_3d = np.expand_dims(normalized_sequence, axis=0)  # เพิ่มมิติเป็น 3D

                # ทำนาย Class
                prediction = model.predict(normalized_sequence_3d)[0]
                predicted_class = classes[np.argmax(prediction)]
            else:
                prediction = [0, 0, 0]
                predicted_class = "Detecting..."

            # อัปเดตภาพใน Matplotlib
            axes[0, 0].cla()
            axes[0, 1].cla()
            axes[0, 2].cla()
            axes[1, 0].cla()
            axes[1, 1].cla()
            axes[1, 2].cla()

            # Step 1: Input Data Preparation
            axes[0, 0].imshow(np.array(sequence_data).T, aspect='auto', cmap='Blues')
            axes[0, 0].set_title('Step 1: Input Data Preparation')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Features')

            # Step 2: CNN Features (จำลอง)
            cnn_output = np.random.rand(15, 16)  # Output จำลองจาก CNN
            axes[0, 1].imshow(cnn_output.T, aspect='auto', cmap='Greens')
            axes[0, 1].set_title('Step 2: CNN Output')
            axes[0, 1].set_xlabel('Reduced Timesteps')
            axes[0, 1].set_ylabel('Filters')

            # Step 3: LSTM Features (จำลอง)
            lstm_output = np.random.rand(32)  # Output จำลองจาก LSTM
            axes[0, 2].bar(range(len(lstm_output)), lstm_output, color='orange')
            axes[0, 2].set_title('Step 3: LSTM Output')
            axes[0, 2].set_xlabel('LSTM Units')
            axes[0, 2].set_ylabel('Values')

            # Step 4: Real-Time Hand Skeleton
            axes[1, 0].imshow(image[..., ::-1])  # BGR -> RGB
            axes[1, 0].set_title('Step 4: Hand Skeleton')
            axes[1, 0].axis('off')

            # Step 5: Real-Time Class Prediction
            axes[1, 1].bar(classes, prediction, color='purple')
            axes[1, 1].set_title('Step 5: Class Prediction')
            axes[1, 1].set_ylabel('Probability')

            # Step 6: Overlay Feedback on Video
            axes[1, 2].imshow(image[..., ::-1])  # BGR -> RGB
            axes[1, 2].set_title(f'Step 6: Feedback - {predicted_class}')
            axes[1, 2].axis('off')

            plt.pause(0.01)

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็น RGB และส่งเข้า Mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # วาดโครงกระดูกมือบนภาพและประมวลผล
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ประมวลผลและแสดงผล
    process_frame(image, results)

    # แสดงผลใน OpenCV
    cv2.imshow('Hand Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == 27:  # กด 'Esc' เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
plt.close()
