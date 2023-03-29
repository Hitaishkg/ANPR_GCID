import cv2
def extract_plate(): # the function detects and perfors blurring on the number plate.
	img= cv2.imread(r"/home/vishnu/Documents/ANMR_GCID/images/Cars0.png")
	plate_img=img.copy()
	#Loads the data required for detecting the license plates from cascade classifier.
	plate_cascade = cv2.CascadeClassifier('/home/vishnu/Documents/ANMR_GCID/model/numberplatemodel.xml')

	# detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)

	for (x,y,w,h) in plate_rect:
		a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1])) #parameter tuning
		plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
		# finally representing the detected contours by drawing rectangles around the edges.
		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
        
	return plate_img, plate # returning the processed image.

# cv2.imshow('hb',img)
processed_img, plate_img = extract_plate()
cv2.imshow('Processed Image', processed_img)
cv2.imshow('License Plate', plate_img)
cv2.imwrite("images/lisence_plate.jpg", plate_img)  
cv2.waitKey(0)
cv2.destroyAllWindows()