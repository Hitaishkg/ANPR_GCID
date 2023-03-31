import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2, numpy as np

# Set the folder paths
train_dir = '../Datasets/train/'
test_dir = '../Datasets/test/'

# Define the image data generators with data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Define the image data generator without data augmentation for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training set images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical')

# Load the test set images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical')

# Define the OCR model architecture
model = tf.keras.models.Sequential([
    # Convolutional layers for feature extraction
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    
    # LSTM layer for sequence recognition
    tf.keras.layers.Reshape((4, 128)),  # Reshape output from conv layers to fit LSTM layer
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    
    # Output layer
    tf.keras.layers.Dense(62, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)


model = tf.keras.models.load_model('path/to/trained/model.h5')

# Load the test image
img = cv2.imread('../Datasets/img1.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32))  # Resize image to fit model input shape
img = np.expand_dims(img, axis=-1)  # Add batch dimension

# Make the prediction
prediction = model.predict(np.array([img]))


def decode_sequence(sequence, charset):
    charset = [''] + charset  # Add blank character at the beginning
    text = ''
    for i in range(len(sequence)):
        if sequence[i] != 0 and (not (i > 0 and sequence[i - 1] == sequence[i])):
            text += charset[sequence[i]]
    return text

# Decode the predicted text
predicted_text = decode_sequence(prediction)


print(predicted_text)
print(prediction)