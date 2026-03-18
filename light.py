import cv2
import numpy as np

def process_frame(frame):
    """
    Phát hiện và phân loại màu đèn giao thông (đỏ / xanh) trong vùng đèn tín hiệu.
    
    Công dụng chính:
      - Chuyển sang không gian HSV → tạo mask cho màu đỏ và xanh
      - Tìm contour lớn nhất (giả định là đèn)
      - Xác định đèn là RED hay GREEN dựa trên việc mask xanh có pixel hay không
      - Vẽ trực tiếp bounding box + nhãn lên frame
    
    Trả về:
      - frame đã được vẽ annotation đèn
      - chuỗi "RED" hoặc "GREEN" (hoặc None nếu không phát hiện)
    
    Lưu ý: Hàm giả định đèn nằm ở bên phải frame (cx < 915 mới xử lý)
    """
    # Chuyển sang không gian màu HSV để dễ tách màu
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Phạm vi màu xanh lá (có thể cần điều chỉnh tùy camera/video)
    mask_g = cv2.inRange(hsv, (58, 97, 222), (179, 255, 255))
    # Phạm vi màu đỏ (phần dưới của hue)
    mask_r = cv2.inRange(hsv, (0, 43, 184), (56, 132, 255))

    # Kết hợp hai mask
    mask = cv2.bitwise_or(mask_g, mask_r)

    # Tìm các vùng liên thông (contour)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_out = None

    for c in cnts:
        # Bỏ qua các vùng quá nhỏ (nhiễu)
        if cv2.contourArea(c) < 50:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2

        # Bỏ qua các vùng nằm quá bên phải (ngoài vùng đèn giao thông)
        if cx >= 915:
            continue

        # Quyết định màu đèn: ưu tiên kiểm tra mask xanh
        if cv2.countNonZero(mask_g[y:y+h, x:x+w]) > 0:
            label = "GREEN"
            color = (0, 255, 0)
        else:
            label = "RED"
            color = (0, 0, 255)

        label_out = label

        # Vẽ trực tiếp lên frame để debug / hiển thị
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, label_out