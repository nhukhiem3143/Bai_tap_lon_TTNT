import cv2
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from tracker import Tracker
from light import process_frame

model = YOLO("yolov10s.pt")        
names = model.names

tracker = Tracker()                 
cap = cv2.VideoCapture("tr.mp4")      

# Tạo thư mục lưu kết quả
os.makedirs("output", exist_ok=True)
os.makedirs("output/vi_pham", exist_ok=True)

# Cấu hình video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0, (1020, 600))

# Vùng polygon định nghĩa "vạch dừng" / khu vực cấm đi khi đèn đỏ
area = [(324, 313), (283, 374), (854, 392), (864, 322)]

# Tập hợp lưu các ID xe đã vi phạm (để không lưu ảnh nhiều lần)
violation_ids = set()

# ===== VÒNG LẶP XỬ LÝ VIDEO =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))

    # 1. Phát hiện & phân loại màu đèn giao thông
    frame, light = process_frame(frame)
    light = str(light).upper() if light else None

    # 2. Phát hiện các phương tiện bằng YOLO
    results = model(frame, conf=0.4)[0].boxes.data.cpu().numpy()

    detections = []
    for r in results:
        x1, y1, x2, y2, _, cls = r
        cls = int(cls)
        label = names[cls]

        # Chỉ giữ các lớp là phương tiện giao thông
        if label in ["car", "truck", "bus", "motorcycle"]:
            detections.append([int(x1), int(y1), int(x2), int(y2), label])

    # 3. Theo dõi (tracking) các phương tiện qua các frame
    tracked = tracker.update(detections)

    # Xử lý từng đối tượng đã được gán ID
    for x1, y1, x2, y2, id, label in tracked:
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # Kiểm tra tâm đối tượng có nằm trong vùng polygon cấm không
        inside = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)

        # 4. Phát hiện vi phạm: nằm trong vùng + đèn đỏ
        if inside >= 0 and light == "RED":
            if id not in violation_ids:
                violation_ids.add(id)

                filename = f"output/vi_pham/{id}_{label}_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)

                print(f"VI PHAM: {label} ID {id}")

        # Chọn màu hiển thị: đỏ nếu đã vi phạm, xanh nếu chưa
        color = (0,0,255) if id in violation_ids else (0,255,0)

        # Vẽ bounding box & ID
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{label} #{id}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 5. Vẽ giao diện người dùng
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255,255,255), 2)

    cv2.putText(frame, f"Den: {light}", (750,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.putText(frame, f"Vi Pham: {len(violation_ids)} xe", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # 6. Ghi frame vào video kết quả
    out.write(frame)

    # 7. Hiển thị realtime
    cv2.imshow("He thong phat hien vi pham giao thong", frame)

    if cv2.waitKey(1) == 27:    # phím ESC để thoát
        break

# ===== KẾT THÚC =====
cap.release()
out.release()
cv2.destroyAllWindows()