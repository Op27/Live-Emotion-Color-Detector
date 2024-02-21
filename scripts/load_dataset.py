import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the dataset directories
train_dir = 'C:/Users/owner/Documents/my_project/image-classification/FER-2013/train'
test_dir = 'C:/Users/owner/Documents/my_project/image-classification/FER-2013/test'

# Create an instance of ImageDataGenerator for data augmentation (optional)
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

test_datagen = ImageDataGenerator(rescale=1./255) # Only rescaling for validation data

# Load images from directories
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Resize images to 224x224
        batch_size=32,
        color_mode='rgb',  # Load images in RGB
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),  # Resize images to 224x224
        batch_size=32,
        color_mode='rgb',  # Load images in RGB
        class_mode='categorical')

print("Training and test sets loaded successfully.")
