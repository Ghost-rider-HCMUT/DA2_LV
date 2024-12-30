import re
import cv2
import numpy as np
import pytesseract

# Load image
img = cv2.imread('image/1.jpg')
cv2.imshow('Original', img)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# Adaptive Thresholding 
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find largest rectangle (license plate)
largest_rectangle = [0, 0]
for cnt in contours:
    lenght = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, lenght, True)
    if len(approx) == 4:  # Assuming a rectangle (license plate)
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]

x, y, w, h = cv2.boundingRect(largest_rectangle[1])
cv2.drawContours(img, [largest_rectangle[1]], 0, (0, 255, 0), 6)
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
warped = cv2.resize(warped, (width * 1, height * 1))

# Convert to grayscale and threshold for OCR
gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_warped, (3,3), 0)
thresh_warped = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh_warped = thresh_warped[2:-2, 2:-2]  # Crop a little bit to remove borders

# Show cropped threshold image
cv2.imshow('Cropped after rotation', thresh_warped)

# Morphological operation to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh_warped, cv2.MORPH_OPEN, kernel, iterations=0)
invert = 255 - opening  # Invert the image

# Show the inverted image
cv2.imshow('Inverted Image', invert)

# OCR with Tesseract
data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6 --oem 3')
data = re.sub(r'[^A-Z0-9]', '', data)
print("Bien so xe la:")
print(data)

# Now combine all the images into one display
# Create a list of images to show
images_to_show = [img, warped, thresh_warped, invert]

# Resize images to have the same height (optional for better visual)
height = 400  # You can change the height as needed
resized_images = []

# Ensure all images have 3 channels (color) by converting grayscale images to BGR
for image in images_to_show:
    # If the image has 2 dimensions (grayscale), convert it to 3 channels (BGR)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized_image = cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))
    resized_images.append(resized_image)

# Concatenate the images horizontally
# Ensure each row has up to 4 images, then down to next row if needed
rows = []
for i in range(0, len(resized_images), 4):
    row_images = resized_images[i:i+4]  # Take 4 images at a time
    row = np.hstack(row_images)  # Combine them horizontally
    rows.append(row)

# Stack rows vertically to create the final image
final_image = np.vstack(rows)

# Show the final image
cv2.imshow('All Images Combined', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
