import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
from keras.preprocessing.image import img_to_array

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Path to the dataset
train_fake_path = r'M:\ml projects\Externship-Fake-Real-Logo-Detection\dataset\train\Fake'
train_genuine_path = r'M:\ml projects\Externship-Fake-Real-Logo-Detection\dataset\train\Genuine'
test_fake_path = r'M:\ml projects\Externship-Fake-Real-Logo-Detection\dataset\test\Fake'
test_genuine_path = r'M:\ml projects\Externship-Fake-Real-Logo-Detection\dataset\test\Genuine'

# Assuming all images have the same size, adjust if necessary
img_size = (128, 128)

# Load and preprocess images
def load_and_preprocess(folder_path, label):
    images = []
    labels = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                img = Image.open(img_path).resize(img_size)
                img_array = np.array(img)
                img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return images, labels

# Load training data
try:
    train_fake_images, train_fake_labels = load_and_preprocess(train_fake_path, 0)  # Assuming 0 for fake
    train_genuine_images, train_genuine_labels = load_and_preprocess(train_genuine_path, 1)  # Assuming 1 for genuine

    # Concatenate fake and genuine data
    X_train = np.concatenate([train_fake_images, train_genuine_images], axis=0)
    y_train = np.concatenate([train_fake_labels, train_genuine_labels], axis=0)

    # Convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    # Define the CNN model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_fake_images, test_fake_labels = load_and_preprocess(test_fake_path, 0)
    test_genuine_images, test_genuine_labels = load_and_preprocess(test_genuine_path, 1)

    X_test = np.concatenate([test_fake_images, test_genuine_images], axis=0)
    y_test = np.concatenate([test_fake_labels, test_genuine_labels], axis=0)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred)
    accuracy = accuracy_score(y_test, y_pred_classes)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    # Save the model if needed
    model.save('./output/logo_detection_model.h5')

except FileNotFoundError as e:
    print(f"Directory existence check failed:\n{e}")