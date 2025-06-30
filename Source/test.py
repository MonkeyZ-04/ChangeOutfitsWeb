import cv2
import numpy as np

# กำหนดพิกัดของกรอบสี่เหลี่ยม (x_min, y_min, x_max, y_max)
# bbox_shirt = np.array([60, 240, 630, 650])
# แปลงให้อยู่ในรูปแบบที่ OpenCV ต้องการ (x_min, y_min, width, height)
# โดย width = x_max - x_min และ height = y_max - y_min
x_min, y_min, x_max, y_max = 0, 350, 660, 650
width = x_max - x_min
height = y_max - y_min
bbox_shirt_opencv_format = (x_min, y_min, width, height)


# 1. โหลดรูปภาพ
# ให้แน่ใจว่าคุณมีไฟล์รูปภาพชื่อ 'your_image.jpg' ในไดเรกทอรีเดียวกันกับสคริปต์
# หรือระบุพาธแบบเต็มไปยังรูปภาพของคุณ
image_path = 'OUTFITS/2.5.jpg' # *** เปลี่ยนชื่อไฟล์นี้เป็นชื่อไฟล์รูปภาพของคุณ ***
img = cv2.imread(image_path)

# ตรวจสอบว่าโหลดรูปภาพได้สำเร็จหรือไม่
if img is None:
    print(f"Error: Could not load image from {image_path}")
    print("Please make sure the image file exists and the path is correct.")
else:
    # กำหนดค่าสีของเส้นกรอบ (BGR: Blue, Green, Red) และความหนาของเส้น
    # ในที่นี้ใช้สีแดง (0, 0, 255) และความหนา 2 พิกเซล
    color = (0, 0, 255)  # สีแดง
    thickness = 2        # ความหนาของเส้น

    # วาดกรอบสี่เหลี่ยมบนรูปภาพ
    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    # หรือใช้รูปแบบที่แปลงไว้แล้ว
    # img = cv2.rectangle(img, bbox_shirt_opencv_format, color, thickness)
    
    # แยกพิกัด x, y, width, height
    x, y, w, h = bbox_shirt_opencv_format
    
    # วาดกรอบสี่เหลี่ยม
    # img: รูปภาพที่จะวาด
    # (x, y): พิกัดมุมบนซ้ายของกรอบ
    # (x + w, y + h): พิกัดมุมล่างขวาของกรอบ
    # color: สีของเส้นกรอบ (BGR)
    # thickness: ความหนาของเส้น
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # 3. แสดงรูปภาพพร้อมกรอบสี่เหลี่ยม
    cv2.imshow('Image with Bounding Box', img)

    # รอให้ผู้ใช้กดปุ่มใดๆ เพื่อปิดหน้าต่างแสดงผล
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # (Optional) หากต้องการบันทึกรูปภาพที่วาดกรอบแล้ว
    # cv2.imwrite('image_with_bbox.jpg', img)
    # print("Image with bounding box saved as 'image_with_bbox.jpg'")