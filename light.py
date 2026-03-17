import cv2
import numpy as np

def process_frame(frame):
    # 1. ĐỊNH NGHĨA NGƯỠNG MÀU (HSV)
    # Khoảng màu XANH (Green) trong không gian HSV
    lower_range = np.array([58, 97, 222])   
    upper_range = np.array([179, 255, 255])
    
    # Khoảng màu ĐỎ (Red) trong không gian HSV
    lower_range1 = np.array([0, 43, 184])   
    upper_range1 = np.array([56,132, 255])

    # 2. CHUYỂN ĐỔI KHÔNG GIAN MÀU
    # Chuyển ảnh từ BGR (mặc định OpenCV) sang HSV
    # HSV giúp phân tách màu sắc tốt hơn so với BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. TẠO MASK THEO MÀU
    # Tạo mask cho vùng màu xanh
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Tạo mask cho vùng màu đỏ
    mask1 = cv2.inRange(hsv, lower_range1, upper_range1)

    # 4. KẾT HỢP CÁC MASK
    # Gộp 2 mask (đỏ + xanh) thành 1 mask chung
    combined_mask = cv2.bitwise_or(mask, mask1)

    # 5. NHỊ PHÂN HÓA (LÀM SẠCH MASK)
    # Chuyển mask về dạng nhị phân rõ ràng (0 hoặc 255)
    _, final_mask = cv2.threshold(combined_mask, 254, 255, cv2.THRESH_BINARY)

    # Biến lưu kết quả nhận diện cuối cùng
    detected_label = None

    # 6. TÌM CONTOUR (ĐỐI TƯỢNG)
    cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in cnts:
        if cv2.contourArea(c) > 50:
            
            # Lấy hình chữ nhật bao quanh contour
            x, y, w, h = cv2.boundingRect(c)
            # 7. TÍNH TÂM ĐỐI TƯỢNG
            
            cx = x + w // 2   # Tọa độ x của tâm
            cy = y + h // 2   # Tọa độ y của tâm
            
            #  giới hạn khu vực đèn
            if cx < 915:
             
                # Nếu trong vùng này có pixel màu xanh
                if cv2.countNonZero(mask[y:y+h, x:x+w]) > 0:
                    color = (0, 255, 0)      # Màu vẽ khung (xanh)
                    text_color = (0, 255, 0)
                    label = "GREEN"
                
                # Nếu có pixel màu đỏ
                elif cv2.countNonZero(mask1[y:y+h, x:x+w]) > 0:
                    color = (0, 0, 255)      # Màu vẽ khung (đỏ)
                    text_color = (0, 0, 255)
                    label = "RED"
            else:
                continue
            detected_label = label 

            # 9. VẼ KẾT QUẢ LÊN ẢNH
            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
            # Vẽ tâm đối tượng (chấm xanh dương)
            cv2.circle(frame, (cx, cy), 1, (255, 0, 0), -1)
            
            # Hiển thị nhãn (RED / GREEN)
            cv2.putText(frame, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
    return frame, detected_label