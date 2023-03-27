#code templte for image enhancement 
import tensorflow as tf
from tensorflow import keras
import os
import glob
input_folder="/home/vishnu/Documents/ANMR_GCID/images"
i=0
# Load low-resolution images
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        low_res_images = "/home/vishnu/Documents/ANMR_GCID/horizontal/output"+str(i)+".png"
        high_res_images = "/home/vishnu/Documents/ANMR_GCID/images/Cars"+str(i)+".jpg"
        low_res_images = tf.image.resize(low_res_images, [256, 256])
        high_res_images = tf.image.resize(high_res_images, [256, 256])
        i=i+1


# for image in glob.glob(f"{low_res_images}/*.jpg"):

# Resize images to a consistent size

# Define the model architecture
model = keras.models.Sequential([
  # Add convolutional layers with ReLU activation
  keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
  keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
  keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
  keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid'),
])

# Compile the model with a mean squared error loss function and an Adam optimizer
model.compile(loss='mse', optimizer='adam')


# Train the model on the low-resolution and high-resolution image pairs
model.fit(low_res_images, high_res_images, epochs=10, batch_size=16)

# Load a low-resolution image
low_res_image = "/home/vishnu/Documents/ANMR_GCID/horizontal"

# Resize the image to a consistent size
low_res_image = tf.image.resize(low_res_image, [256, 256])

# Generate a high-resolution image from the low-resolution input using the trained model
high_res_image = model.predict(tf.expand_dims(low_res_image, axis=0))

# Post-process the generated high-resolution image to remove any artifacts or noise
high_res_image = tf.clip_by_value(high_res_image, 0, 1)
high_res_image = tf.image.convert_image_dtype(high_res_image[0], dtype=tf.uint8)

