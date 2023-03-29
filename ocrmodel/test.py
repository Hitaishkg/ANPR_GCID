from anpr import PyImageSearchANPR
from imutils import paths
import argparse
import imutils
import cv2
def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()
anpr = None
iteration = 0
input_dir = 'images/ts1.jpeg'
clear_border = -1
psm = 7
debug = -1
algorithm = 1
save = 1
morphology = 'bh'
input_dir_name = input_dir
anpr = PyImageSearchANPR(debug=debug > 0)
# grab all image paths in the input directory
# imagePaths = sorted(list(paths.list_images(input_dir)))
# imagePaths = sorted(list(paths.list_images(input_dir)))

	# load the input image from disk and resize it
image = cv2.imread(input_dir)

image = imutils.resize(image, width=600)
	# apply automatic license plate recognition
(lpText, lpCnt) = anpr.find_and_ocr(image, psm=psm,
		clearBorder=clear_border > 0)
	# only continue if the license plate was successfully OCR'd
if lpText is not None and lpCnt is not None:
		# fit a rotated bounding box to the license plate contour and
		# draw the bounding box on the license plate
	box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
	box = box.astype("int")
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
		# compute a normal (unrotated) bounding box for the license
		# plate and then draw the OCR'd license plate text on the
		# image
	(x, y, w, h) = cv2.boundingRect(lpCnt)
	cv2.putText(image, cleanup_text(lpText), (x, y - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		# show the output ANPR image
	print("[INFO] {}".format(lpText))
		# cv2.imshow("Output ANPR", image)
	cv2.imwrite("images/vis.jpej",image )
	cv2.waitKey(0)