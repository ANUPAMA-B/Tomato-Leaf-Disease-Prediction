import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model_save_path = r"model.h5"
model = load_model(model_save_path)

# Data Preprocessing (same as during training)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load validation data (same as used during training)
validation_generator = test_datagen.flow_from_directory(
    "tomato/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Don't shuffle to keep correspondence between images and labels
)

# Get class labels
class_labels = list(validation_generator.class_indices.keys())

# Predict on validation set
predictions = model.predict(validation_generator, steps=len(validation_generator), verbose=1)

# Get the predicted class indices
predicted_class_indices = np.argmax(predictions, axis=1)

# Get the true class indices
true_class_indices = validation_generator.classes

# Print the expected (true) and predicted class names
for i in range(len(true_class_indices)):
    true_label = class_labels[true_class_indices[i]]
    predicted_label = class_labels[predicted_class_indices[i]]
    print(f"Expected: {true_label}, Predicted: {predicted_label}")
