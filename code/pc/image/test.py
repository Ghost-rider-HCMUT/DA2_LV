import cv2
import numpy as np
import pytesseract
import re

# Đường dẫn đến ảnh
image_path = 'image/8.jpg'

# Đọc ảnh
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Không thể đọc được ảnh, hãy kiểm tra lại đường dẫn.")

cv2.imshow('Original', img)

# Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# Áp dụng adaptive threshold để tăng cường độ tương phản
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Threshold', thresh)

# Tìm các contours trong ảnh
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Tìm hình kín có 4 cạnh
# Find largest rectangle (license plate)
largest_rectangle = [0, 0, 0]
for cnt in contours:
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:  # Assuming a rectangle (license plate)
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]

x, y, w, h = cv2.boundingRect(largest_rectangle[1])
cv2.drawContours(img, [largest_rectangle[1]], 0, (0, 255, 0), 4)
cv2.imshow('drawContours', img)

# Get the 4 points of the rectangle
pts = largest_rectangle[2]
pts = pts.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")

# Sort points based on their position
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]  # Top-left
rect[2] = pts[np.argmax(s)]  # Bottom-right
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]  # Top-right
rect[3] = pts[np.argmax(diff)]  # Bottom-left

# Calculate width and height for perspective transform
width = int(np.linalg.norm(rect[1] - rect[0]))
height = int(np.linalg.norm(rect[2] - rect[1]))

# Destination points for transformation
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

# Perspective transformation
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(img, M, (width, height))

# Convert to grayscale and threshold for OCR
gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray_warped, (3,3), 0)
thresh_warped = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# thresh_warped = thresh_warped[2:-2, 2:-2]  # Crop a little bit to remove borders

# Show cropped threshold image
cv2.imshow('Cropped after rotation', thresh_warped)

# Đợi phím bất kỳ để đóng tất cả cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()
