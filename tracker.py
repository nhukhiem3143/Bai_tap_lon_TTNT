import math

class Tracker:
    """
    Lớp Tracker đơn giản thực hiện theo dõi (tracking) các đối tượng qua các frame liên tiếp
    bằng cách so sánh khoảng cách tâm (centroid) giữa các detection hiện tại và các đối tượng cũ.
    Sử dụng phương pháp Nearest Neighbor cơ bản + logic "disappeared" để xóa ID khi mất quá lâu.
    """
    def __init__(self, max_dist=70, max_disappeared=15):
        # Lưu trữ thông tin các đối tượng đang được theo dõi
        self.objects = {}           # id -> ((cx, cy), label)
        self.disappeared = {}       # id -> số frame liên tiếp không được phát hiện
        self.id_count = 0           # bộ đếm để tạo ID mới

        # Ngưỡng khoảng cách tối đa để coi 2 tâm là cùng một đối tượng
        self.max_dist = max_dist
        # Số frame tối đa cho phép đối tượng "mất tích" trước khi xóa
        self.max_disappeared = max_disappeared

    def update(self, detections):
        """
        Hàm chính cập nhật tracker với danh sách detection mới của frame hiện tại.
        
        Luồng xử lý:
          1. Duyệt qua từng detection mới → tính tâm → tìm xem có khớp với object cũ nào không (dựa vào khoảng cách)
          2. Nếu khớp → cập nhật vị trí & reset disappeared
          3. Nếu không khớp → tạo ID mới
          4. Với các ID cũ không được khớp frame này → tăng bộ đếm disappeared
          5. Xóa các ID đã disappeared quá ngưỡng
        
        Trả về: danh sách các đối tượng đã được gán ID [x1,y1,x2,y2, id, label]
        """
        results = []
        current_ids = set()         # tập hợp các ID xuất hiện trong frame hiện tại

        # Gán detection mới vào các ID cũ hoặc tạo ID mới
        for x1, y1, x2, y2, label in detections:
            cx, cy = (x1+x2)//2, (y1+y2)//2

            found = False
            for id, ((px, py), _) in self.objects.items():
                # Nếu khoảng cách tâm nhỏ hơn ngưỡng → coi là cùng đối tượng
                if math.hypot(cx-px, cy-py) < self.max_dist:
                    self.objects[id] = ((cx, cy), label)
                    self.disappeared[id] = 0
                    results.append([x1, y1, x2, y2, id, label])
                    current_ids.add(id)
                    found = True
                    break

            # Detection mới không khớp object nào → tạo ID mới
            if not found:
                self.objects[self.id_count] = ((cx, cy), label)
                self.disappeared[self.id_count] = 0
                results.append([x1, y1, x2, y2, self.id_count, label])
                current_ids.add(self.id_count)
                self.id_count += 1

        # Cập nhật trạng thái disappeared cho các object không xuất hiện frame này
        for id in list(self.objects.keys()):
            if id not in current_ids:
                self.disappeared[id] += 1
                if self.disappeared[id] > self.max_disappeared:
                    del self.objects[id]
                    del self.disappeared[id]

        return results