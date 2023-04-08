import tensorflow as tf
import numpy as np
import cv2
import os

# Define the input image size and maximum sequence length
IMAGE_SIZE = (64, 128)
MAX_SEQ_LENGTH = 10

# Define the path to the dataset
DATASET_PATH = "/path/to/dataset/"

# Define the characters that can appear in the text labels
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Load the dataset and preprocess the images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    return image

def create_datasets():
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for file_name in os.listdir(DATASET_PATH):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(DATASET_PATH, file_name)
            label = file_name.split("_")[0]

            # Convert the label to a one-hot encoded vector
            label_vector = np.zeros((len(CHARACTERS)))
            for i, c in enumerate(CHARACTERS):
                if c in label:
                    label_vector[i] = 1.0

            # Preprocess the image and add it to the appropriate dataset
            image = preprocess_image(image_path)
            if "test" in file_name:
                test_images.append(image)
                test_labels.append(label_vector)
            else:
                train_images.append(image)
                train_labels.append(label_vector)

    # Convert the datasets to TensorFlow tensors
    train_images = tf.constant(train_images)
    train_labels = tf.constant(train_labels)
    test_images = tf.constant(test_images)
    test_labels = tf.constant(test_labels)

    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = create_datasets()
# Define the model architecture
def build_model():
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    # Define the convolutional layers for feature extraction
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
# Define the recurrent layers for sequence modeling
    x = tf.keras.layers.Reshape((-1, 64 * 8))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)

# Define the attention mechanism
    attention = tf.keras.layers.Dense(1, activation="tanh")(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation("softmax")(attention)
    attention = tf.keras.layers.RepeatVector(128 * 2)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)

# Apply the attention weights to the sequence output
    x = tf.keras.layers.Multiply()([x, attention])
    x = tf.keras.layers.Lambda(lambda a: tf.keras.backend.sum(a, axis=1))(x)

# Define the output layer
    output_layer = tf.keras.layers.Dense(len(CHARACTERS), activation="softmax")(x)

# Define the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model

BATCH_SIZE = 32
EPOCHS = 10
model=build_model()
# Train the model
history = model.fit(
    train_images,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_images, test_labels),
)
# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Compute the model's predictions on the test dataset
predictions = model.predict(test_images)
predicted_labels = ["".join([CHARACTERS[i] for i in np.argmax(p)]) for p in predictions]
true_labels = ["".join([CHARACTERS[i] for i in np.argmax(l)]) for l in test_labels]

# Compute the precision, recall, and F1 score of the model's predictions
tp = sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == true_labels[i]])
fp = sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] != true_labels[i]])
fn = sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] != true_labels[i]])
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1_score:.4f}")
# Load the trained weights into the model
model.load_weights("model_weights.h5")

# Preprocess the input image
image = cv2.imread("test_image.png")
image = preprocess_image(image)

# Use the model to make predictions on the preprocessed image
prediction = model.predict(np.array([image]))
predicted_label = "".join([CHARACTERS[i] for i in np.argmax(prediction[0])])
print(f"Predicted label: {predicted_label}")