import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ใช้งาน Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# เก็บ Time Series ของ Key Points
temporal_data = []
max_frames = 30  # เก็บข้อมูลสูงสุด 30 เฟรม

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(1)

# ฟังก์ชันสำหรับ Matplotlib
fig, ax = plt.subplots()
lines = []
for _ in range(21):  # มี 21 Key Points
    line, = ax.plot([], [], marker="o")  # เส้นและจุดในแต่ละ Key Point
    lines.append(line)

ax.set_xlim(0, max_frames)
ax.set_ylim(0, 1)  # Normalized ค่า Y อยู่ระหว่าง 0-1
ax.set_title("Temporal Data Creation (Y-axis of Key Points)")
ax.set_xlabel("Frames")
ax.set_ylabel("Y Coordinate (Normalized)")

# ฟังก์ชันอัปเดตข้อมูลสำหรับ Matplotlib
def update(frame):
    global temporal_data
    ret, image = cap.read()
    if not ret:
        return lines

    # แปลงภาพเป็น RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ประมวลผล Mediapipe
    results = hands.process(image_rgb)

    # ตรวจจับ Key Points
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            temporal_data.append(keypoints)

            # จำกัดข้อมูลใน Time Series ไว้ที่ max_frames
            if len(temporal_data) > max_frames:
                temporal_data.pop(0)

            # แยกข้อมูลแกน Y สำหรับทุก Key Point
            temporal_y = np.array([[point[1] for point in frame] for frame in temporal_data])

            # อัปเดตเส้นในกราฟสำหรับแต่ละ Key Point
            for i, line in enumerate(lines):
                if temporal_y.shape[0] > 0:
                    line.set_data(range(temporal_y.shape[0]), temporal_y[:, i])

    return lines

# ใช้ Matplotlib Animation
ani = FuncAnimation(fig, update, interval=30, blit=True)

# แสดงผล
plt.show()

# ปิดกล้องเมื่อเสร็จสิ้น
cap.release()
hands.close()
