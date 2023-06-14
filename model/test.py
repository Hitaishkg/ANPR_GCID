import cv2
from prediction import ocr_text

def extract_plate():
	img= cv2.imread(r"/home/vishnu/Documents/ANMR_GCID/model/images/ts1.jpeg")
	# img.resize((450,450))
	plate_img=img.copy()
	plate_cascade = cv2.CascadeClassifier('/home/vishnu/Documents/ANMR_GCID/model/numberplatemodel.xml')
	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.25, minNeighbors = 7)
	for (x,y,w,h) in plate_rect:
		# w=w+30
		a,b = (int(0.013*img.shape[0]), int(0.028*img.shape[1])) 
		plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
	return plate_img, plate

processed_img, plate_img = extract_plate()
cv2.imshow('Processed Image', processed_img)
cv2.imshow('License Plate', plate_img)
cv2.imwrite("model/images/lisence_plate8.jpg", plate_img)  
cv2.imwrite("model/images/detection.png",processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_path='/home/vishnu/Documents/ANMR_GCID/model/images/lisence_plate8.jpg'
final_output=ocr_text(image_path)
print(final_output)
# except:
# 	image_path='/home/vishnu/Documents/ANMR_GCID/model/images/ts1.jpeg'
# 	final_output=ocr_text(image_path)
# 	print(final_output)