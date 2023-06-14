import cv2
import numpy as np

# Load the low-resolution license plate image
img = cv2.imread('/home/vishnu/Documents/ANMR_GCID/model/images/tv1.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to improve contrast
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply morphological operations to remove small noise and fill gaps in letters/numbers
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Apply edge detection to highlight the contours of the license plate characters
edges = cv2.Canny(morph, 50, 150)

# Find contours in the image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image for visual confirmation
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Display the final processed image
cv2.imshow('Processed Image', img)
cv2.imwrite('finalimg.jpeg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()