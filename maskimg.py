import cv2

# Load the two images to be masked
img1 = cv2.imread("C:/Users/hitai/Downloads/photos/threshold.jpeg.jpg")

conv = cv2.bitwise_not(img1)
# img1_bgr = cv2.cvtColor(conv, cv2.COLOR_GRAY2BGR)
print (conv).shape
print(conv).size 
cv2.imshow("NOT",conv)
cv2.waitKey(0)