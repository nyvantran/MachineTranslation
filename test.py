import requests
import cv2
import numpy as np

# --- THAY ĐỔI URL NÀY ---
url = "http://192.168.1.20/stream"
# -------------------------

try:
    print(f"Bắt đầu kết nối tới stream: {url}")
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()

        image_bytes = bytes()

        for chunk in response.iter_content(chunk_size=4096):  # Tăng chunk size có thể cải thiện hiệu năng
            if not chunk:
                # Bỏ qua các chunk rỗng nếu có
                continue

            image_bytes += chunk

            # --- LOGIC ĐÃ SỬA LỖI ---
            # 1. TÌM ĐIỂM BẮT ĐẦU CỦA FRAME
            start = image_bytes.find(b'\xff\xd8')

            # 2. NẾU TÌM THẤY ĐIỂM BẮT ĐẦU, TÌM TIẾP ĐIỂM KẾT THÚC KỂ TỪ ĐÓ
            #    Điều này đảm bảo 'end' luôn nằm sau 'start'
            if start != -1:
                end = image_bytes.find(b'\xff\xd9', start)  # Tìm 'end' SAU 'start'

                # 3. KIỂM TRA ĐIỀU KIỆN: PHẢI TÌM THẤY CẢ HAI VÀ END PHẢI SAU START
                if end != -1:
                    print(start, end)
                    # Cắt lấy frame hoàn chỉnh
                    jpeg_frame = image_bytes[start:end + 2]

                    # Cắt bỏ frame vừa xử lý khỏi buffer để xử lý frame tiếp theo
                    image_bytes = image_bytes[end + 2:]

                    # Giải mã và hiển thị frame
                    # Thêm kiểm tra len(jpeg_frame) > 0 để chắc chắn 100%
                    if len(jpeg_frame) > 0:
                        frame = cv2.imdecode(np.frombuffer(jpeg_frame, dtype=np.uint8), cv2.IMREAD_COLOR)

                        if frame is not None:
                            cv2.imshow('Video Stream', frame)

            # Chờ 1ms, nếu nhấn phím 'q' thì thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except requests.exceptions.RequestException as e:
    print(f"Lỗi kết nối: {e}")
except KeyboardInterrupt:
    print("\nĐã dừng chương trình.")
finally:
    cv2.destroyAllWindows()
    print("Đã đóng kết nối và cửa sổ.")
