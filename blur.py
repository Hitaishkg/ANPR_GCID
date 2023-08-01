import cv2
import numpy as np
from PIL import Image, ImageFilter
import os
from wand.image import Image as im
i=0
input_folder="images/"
# for i in range(100):
    # image_path = "/home/mahima/Documents/deep learning/images/Cars0.png"
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_file = Image.open(os.path.join(input_folder, filename))
# image_file = Image.open('/home/mahima/Documents/deep learning/images/Cars0.png')
        # image_file = Image.open(image)
        rgb=image_file.convert('RGB')

        rgb.save("image_name"+str(i)+".jpg", quality=1)
            # x='/home/mahima/Documents/deep learninyyg/image_name3.jpg'
        image_blur= Image.open("/home/mahima/Documents/deep learning/image_name"+str(i)+".jpg")
            # image_blur.show()

            # blurImage = image_blur.filter(ImageFilter.BLUR)
            # image_blur.save("img_name"+str(i)+".jpg")
            # image_noise=Image.open('/home/mahima/Documents/deep learning/image_name4.jpg')
        blur_factor=3
        image_blur = image_blur.filter(ImageFilter.GaussianBlur(blur_factor))
        noise_factor=0.5
        noise = Image.effect_noise(image_blur.size, noise_factor)


        with im(filename ="image_name"+str(i)+".jpg") as img:
            img.noise("laplacian", attenuate = 100.0)
            img.save(filename ="noise"+str(i)+".jpg")
        # img = cv2.imread("noise.jpg")
        # psf = np.zeros((50, 50, 3))
        # psf = cv2.ellipse(psf, 
        #                   (25, 25), # center
        #                   (22, 0), # axes -- 22 for blur length, 0 for thin PSF 
        #                   15, # angle of motion in degrees
        #                   0, 360, # ful ellipse, not an arc
        #                   (1, 1, 1), # white color
        #                   thickness=-1) # filled

        # psf /= psf[:,:,0].sum() # normalize by sum of one channel 
        #                         # since channels are processed independently
        # imfilt = cv2.filter2D(img, -1, psf)
        # # img.save("image1.jpg")
        img = cv2.imread("noise"+str(i)+".jpg")
        
        # Specify the kernel size.
        # The greater the size, the more the motion.
        kernel_size = 17
        
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
        vertical_mb = cv2.filter2D(img, -1, kernel_v)z
        
        # Apply the horizontal kernel.
        horizonal_mb = cv2.filter2D(img, -1, kernel_h)
        
        # Save the outputs.
        # cv2.imwrite('car_vertical.jpg', vertical_mb)
        cv2.imwrite("/home/mahima/Documents/deep learning/horizontal/output"+str(i)+".jpg", horizonal_mb)
        i=i+1


