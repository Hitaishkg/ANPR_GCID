import cv2
from PIL import Image
from prediction import ocr_text
import numpy as np

image= cv2.imread(r"/home/vishnu/Documents/ANMR_GCID/model/images/anish3.jpg")
car_img=image.copy()
# image = image.resize((450,250))
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
Image.fromarray(grey)
blur = cv2.GaussianBlur(grey,(5,5),0)
Image.fromarray(blur)
dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
Image.fromarray(closing)
# cv2.imshow('img',closing)

car_cascade = cv2.CascadeClassifier('/home/vishnu/Documents/ANMR_GCID/model/carmodel.xml')
cars = car_cascade.detectMultiScale(closing, 1.1, 1)

for (x,y,w,h) in cars:
    final = image[y:y+h, x:x+w, :]
    cv2.rectangle(car_img,(x,y),(x+w,y+h),(255,0,0),2)

# print(" cars found")

# Image.fromarray(image_arr)
cv2.imshow('carimg',car_img)
cv2.imshow('cimg',final)
cv2.waitKey(0)
cv2.destroyAllWindows()