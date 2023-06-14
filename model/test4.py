import cv2
from prediction import ocr_text
from google_cloudocr import google_ocr
def extract_plate(img): # the function detects and perfors blurring on the number plate.
	plate_img = img.copy()
	
	#Loads the data required for detecting the license plates from cascade classifier.
	plate_cascade = cv2.CascadeClassifier(r"/home/vishnu/Documents/ANMR_GCID/model/numberplatemodel2.xml")

	# detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)
	#scale factor values can range from 1.02 to 1.3 or 1.5 usually 
	for (x,y,w,h) in plate_rect:
		a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1])) #parameter tuning
		plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
		# finally representing the detected contours by drawing rectangles around the edges.
		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
        
	return plate_img, plate # returning the processed image.
try:
	path='/home/vishnu/Documents/ANMR_GCID/model/images/ts1.jpeg'
	img = cv2.imread(path)
	processed_img, plate_img = extract_plate(img)
	cv2.imshow('Processed Image', processed_img)
	cv2.imshow('License Plate', plate_img)
	cv2.imwrite("model/images/lisence_plate5.jpg", plate_img)  
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	image_path='/home/vishnu/Documents/ANMR_GCID/model/images/ts1.jpeg'
	output1=google_ocr(image_path)
	print(output1)
	final_output=ocr_text(image_path)
	print(final_output)	
except:
	output1=google_ocr(path)
	print(output1)
	final_output=ocr_text(path)
	print(final_output)