import math

class Tracker:
    def __init__(self):
        # Lưu tọa độ tâm (center) của các đối tượng theo ID
        self.center_points = {}
        # Biến đếm ID, mỗi đối tượng mới sẽ được gán ID tăng dần
        self.id_count = 0

    def update(self, objects_rect):
        # Danh sách kết quả gồm [x, y, w, h, id]
        objects_bbs_ids = []

        # Duyệt qua từng bounding box đầu vào
        for rect in objects_rect:
            x, y, w, h = rect

            # Tính tọa độ tâm của bounding box
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Giả sử chưa tìm thấy đối tượng cũ trùng
            same_object_detected = False

            # So sánh với các đối tượng đã lưu trước đó
            for id, pt in self.center_points.items():
                # Tính khoảng cách Euclid giữa 2 tâm
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # Nếu khoảng cách nhỏ hơn ngưỡng → cùng một đối tượng
                if dist < 35:
                    # Cập nhật lại vị trí tâm mới
                    self.center_points[id] = (cx, cy)
                    # Thêm vào danh sách kết quả với ID cũ
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Nếu không trùng với đối tượng nào → tạo ID mới
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Tạo dictionary mới chỉ chứa các ID còn tồn tại
        new_center_points = {}

        for obj_bb_id in objects_bbs_ids:
            # Lấy ID của đối tượng
            _, _, _, _, object_id = obj_bb_id
            # Lấy lại tâm tương ứng
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Cập nhật lại danh sách center_points (loại bỏ ID không còn xuất hiện)
        self.center_points = new_center_points.copy()

        # Trả về danh sách bounding box kèm ID
        return objects_bbs_ids