import cv2

# Load the two images to be masked
img1 = cv2.imread("C:/Users/hitai/Downloads/photos/WhatsApp Image 2023-03-06 at 15.55.50.jpg")
img2 = cv2.imread("C:/Users/hitai/Downloads/photos/threshold.jpeg.jpg")

# Create a binary mask for the first image

rgb_img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)


# Convert the grayscale image to an RGB image


# Apply the mask to the first image
masked_rgb_img1 = cv2.bitwise_and(img, rgb_img1, mask=mask)

# Apply the mask to the second image
masked_img2 = cv2.bitwise_and(rgb_img2, rgb_img2, mask=mask)

# Combine the two masked images
final_image = cv2.add(masked_rgb_img1, masked_img2)

# Display the final image
cv2.imshow('Masked Images', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
