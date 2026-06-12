from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the model
model_path = r"model.h5"

# Load the model
model = load_model(model_path)

# Define the path to your test dataset
test_data_dir = r"tomato\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"  # Update this path with the location of your test dataset

# Set image dimensions (match these with your model input)
img_height = 224  # Example height, update as per your model input
img_width = 224   # Example width, update as per your model input
batch_size = 32   # Set the batch size

# Preprocessing the test images (adjust the rescale as per the dataset)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the test images
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Adjust if your problem is binary ('binary') or multi-class ('categorical')
)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_generator)

# Print the accuracy
print(f"Model accuracy: {accuracy * 100:.2f}%")
