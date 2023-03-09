
import os
import cv2
import numpy as np

# Define the input and output folders
input_folder = "/home/vybhv/Downloads/archive/Indian_Number_Plates/Sample_Images"
output_folder = "/home/vybhv/Downloads/reducto"

# Define the parameters for Gaussian blur and Gaussian noise
kernel_size = (3, 3)
std_dev = 0.5 
mean = 0

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Load the image
    img = cv2.imread(os.path.join(input_folder, filename))

    # Get the original resolution of the image    
    height, width = img.shape[:2]

    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img, kernel_size, 0)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, (height, width, 3)) 

    # Add the noise to the image
    img_noisy = cv2.add(img_blur, noise)

    # Save the processed image to the output folder with the same resolution as the original image
    cv2.imwrite(os.path.join(output_folder, filename), img_noisy, [cv2.IMWRITE_JPEG_QUALITY, 100])

    kernel_size = 30
  
# Create the vertical kernel.
kernel_v = np.zeros((kernel_size, kernel_size))
  
# Create a copy of the same for creating the horizontal kernel.
kernel_h = np.copy(kernel_v)
  
# Fill the middle row with ones.
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
  
# Normalize.
kernel_v /= kernel_size
kernel_h /= kernel_size
  
# Apply the vertical kernel.
vertical_mb = cv2.filter2D(img, -1, kernel_v)
  
# Apply the horizontal kernel.
horizonal_mb = cv2.filter2D(img, -1, kernel_h)
  
# Save the outputs.
cv2.imwrite('car_vertical.jpg', vertical_mb)
cv2.imwrite('car_horizontal.jpg', horizonal_mb)