from PIL import Image, ImageFilter, ImageOps
import os

# prompt user for input and output folder paths
input_folder = input("Enter input folder path: ")
output_folder = input("Enter output folder path: ")

# create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# prompt user for desired resolution
# width = int(input("Enter desired width (pixels): "))
# height = int(input("Enter desired height (pixels): "))

# prompt user for desired amount of noise
noise_factor = float(input("Enter desired noise factor (0-1): "))

# prompt user for desired degree of blur
blur_factor = int(input("Enter desired blur factor (1-10): "))

# loop through images in input folder and modify
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # open image and modify
        image = Image.open(os.path.join(input_folder, filename))

        # reduce resolution
        image = image.resize((276, 182))

        from PIL import Image

        # add noise
        if noise_factor > 0:
           # image = image.convert('L')
           # image = ImageOps.autocontrast(image)
            noise = Image.effect_noise(image.size, noise_factor)
         #   image = Image.composite(image, noise, image)

        # add blur
        if blur_factor > 0:
            image = image.filter(ImageFilter.GaussianBlur(blur_factor))

        # save modified image to output folder
        image.save(os.path.join(output_folder, filename), quality = 25)



#         # Open the image by specifying the image path.
# image_path = "image_name.jpeg"
# image_file = Image.open(image_path)
  
# # the default
# image_file.save("image_name.jpg", quality=95)
  
# # Changing the image resolution using quality parameter
# # Example-1
# image_file.save("image_name2.jpg", quality=25)
  
# # Example-2
# image_file.save("image_name3.jpg", quality=1)