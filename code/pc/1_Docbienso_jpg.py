# LOAD THU VIEN VA MODUL CAN THIET
import re
import cv2
import pytesseract
#DOC HINH ANH - TACH HINH ANH NHAN DIEN
img = cv2.imread('image/8.jpg')
cv2.imshow('Image', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
contours,h = cv2.findContours(thresh,1,2)
largest_rectangle = [0,0]
for cnt in contours:
    lenght = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, lenght, True)
    if len(approx)==4: 
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
x,y,w,h = cv2.boundingRect(largest_rectangle[1])
image = img[y:y+h, x:x+w]
cv2.drawContours(img,[largest_rectangle[1]],0,(0,0,255),2)

cropped = img[y:y+h, x:x+w]
cv2.imshow('Dinh vi bien so xe', img)

# cv2.drawContours(img,[largest_rectangle[1]],0,(255,255,255),4)
#DOC HINH ANH CHUYEN THANH FILE TEXT
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('Crop', thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening
data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6 --oem 3')
data = re.sub(r'[^A-Z0-9]', '', data)
print("Bien so xe la:")
print(data)
cv2.waitKey()
cv2.destroyAllWindows()