import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Load the saved model
model_save_path = r"model.h5"
model = load_model(model_save_path)

# Data Preprocessing (same as during training)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load validation data (this will provide the class labels)
validation_generator = test_datagen.flow_from_directory(
    "tomato/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Don't shuffle to maintain correspondence
)

# Get class labels from the validation generator
class_labels = list(validation_generator.class_indices.keys())
print("Class labels: ", class_labels)

def predict_disease(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    image = img_to_array(image)  # Convert to numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Rescale the image if used during training

    # Predict the class
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability

    # Map the predicted index to the class label (disease name)
    predicted_label = class_labels[predicted_class_index]
    
    return predicted_label

# Example usage
image_path = r"tomato\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train\Tomato___Early_blight\1c71757d-d351-4016-b5bb-06a63d0f2e24___RS_Erly.B 6388.JPG"  # Path to your image
predicted_disease = predict_disease(image_path)
print(f"Predicted Disease: {predicted_disease}")
