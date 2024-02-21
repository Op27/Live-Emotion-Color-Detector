import tensorflow as tf
import sys

sys.path.append('C:/Users/owner/Documents/my_project/image-classification/Basic_CNNs_TensorFlow2.0')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming ResNet18 is correctly referenced within the cloned repository
from resnet import ResNet18



def load_data():
    # Define paths to the dataset directories
    train_dir = 'C:/Users/owner/Documents/my_project/image-classification/original dataset/train'
    test_dir = 'C:/Users/owner/Documents/my_project/image-classification/original dataset/test'

    # Create instances of ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test data

    # Load images from directories and apply preprocessing
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Resize images to match ResNet18 input size
        batch_size=32,
        color_mode='rgb',  # Load images in RGB
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),  # Resize images to match ResNet18 input size
        batch_size=32,
        color_mode='rgb',  # Load images in RGB
        class_mode='categorical'
    )

    return train_generator, test_generator

def build_model():
    # Load the ResNet18 model pre-trained on ImageNet data, excluding its top layer
    base_model = ResNet18(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Define the custom layers to add on top of ResNet18
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # Assuming 7 classes for emotions
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def main():
    train_generator, test_generator = load_data()
    model = build_model()

    # Display the model's architecture
    model.summary()

    # Training the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=25,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size
    )

    # Save the trained model
    model.save('emotion_model_resnet18.h5')

if __name__ == "__main__":
    main()
