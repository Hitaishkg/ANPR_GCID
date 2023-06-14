import cv2

#importing numpy
import numpy as np

#importing pandas to read the CSV file containing our data
import pandas as pd

#importing keras and sub-libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split 

def extract_plate(img): # the function detects and perfors blurring on the number plate.
	plate_img = img
	
	#Loads the data required for detecting the license plates from cascade classifier.
	plate_cascade = cv2.CascadeClassifier(r"C:\Developer\tenserflow object detection\numberplate.xml")

	# detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)

	for (x,y,w,h) in plate_rect:
		a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1])) #parameter tuning
		plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
		# finally representing the detected contours by drawing rectangles around the edges.
		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
        
	return plate_img, plate # returning the processed image

img = cv2.imread(r"C:\Developer\tenserflow object detection\tryfolder\20180402113123_NumberPlate_Swift.jpg")
processed_img, plate_img = extract_plate(img)
cv2.imshow('Processed Image', processed_img)
cv2.imshow('License Plate', plate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()