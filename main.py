import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
import os
from datetime import datetime
from tracker import Tracker 
from light import process_frame # Đảm bảo file light.py có hàm này

# 1. Khởi tạo mô hình phương tiện
model = YOLO("yolov10s.pt")  

# 2. Cấu hình Video
video_path = 'youtube.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output/ket_qua.avi', fourcc, 20.0, (1020, 600))

# 3. Đọc danh sách class COCO
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# 4. Khởi tạo Tracker và vùng kiểm soát
tracker = Tracker()
violation_ids = set() 
area = [(324, 313), (283, 374), (854, 392), (864, 322)] 

# 5. Tạo thư mục lưu ảnh vi phạm theo yêu cầu: output/vi_pham
output_dir = os.path.join('output', 'vi_pham')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (1020, 600))
    
    # Bước A: Nhận diện màu đèn
    processed_frame, detected_label = process_frame(frame)
    detected_label = str(detected_label).upper() 

    # Bước B: Phát hiện xe cộ
    results = model(frame, conf=0.4) 
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a).astype("float")
    
    obj_list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        c = class_list[d]
        if 'car' in c or 'truck' in c or 'bus' in c or 'motorcycle' in c:
            obj_list.append([x1, y1, x2, y2])

    # Bước C: Tracking và kiểm tra vi phạm
    bbox_idx = tracker.update(obj_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        is_inside = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)

        if is_inside >= 0 and detected_label == "RED":
            if id not in violation_ids:
                violation_ids.add(id)
                # Lưu ảnh vào output/vi_pham
                timestamp = datetime.now().strftime('%H-%M-%S')
                img_name = f"ID_{id}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(output_dir, img_name), frame)
                print(f"!!! PHÁT HIỆN VI PHẠM: Đã lưu ảnh xe ID {id} vào {output_dir}")

        # Vẽ khung hiển thị
        color = (0, 0, 255) if id in violation_ids else (0, 255, 0)
        cv2.rectangle(frame, (x3, y3), (x4, y4), color, 2)
        text = f'VIOLATION #{id}' if id in violation_ids else f'ID: {id}'
        cvzone.putTextRect(frame, text, (x3, y3 - 10), 1, 1, colorR=color)

    # Bước D: Vẽ vùng area và thông tin đèn
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 255), 2)
    color_map = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "YELLOW": (0, 255, 255)}
    status_color = color_map.get(detected_label, (127, 127, 127))
    cvzone.putTextRect(frame, f'LIGHT: {detected_label}', (750, 50), 2, 2, colorR=status_color)
    cvzone.putTextRect(frame, f'TOTAL VIOLATIONS: {len(violation_ids)}', (30, 50), 2, 2, colorR=(0, 0, 255))

    output_video.write(frame)
    cv2.imshow("Traffic Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
output_video.release()
cv2.destroyAllWindows()