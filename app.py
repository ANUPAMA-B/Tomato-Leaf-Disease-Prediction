import os
import numpy as np
import pyttsx3
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from werkzeug.utils import secure_filename
from googletrans import Translator

# Initialize Flask app
app = Flask(__name__)


# Directory to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Load model
model_save_path = "model.h5"
model = load_model(model_save_path)

# Data Preprocessing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load validation data to get class labels
validation_generator = test_datagen.flow_from_directory(
    'tomato/New Plant Diseases Dataset(Augmented)/DATASET TOMOTO/valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
)

# Get class labels
class_labels = list(validation_generator.class_indices.keys())
print("Class labels: ", class_labels)

translator = Translator()
def count_uploaded_files():
    """Counts the number of files in the upload folder."""
    return len([name for name in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, name))])


def speak_text(text, lang='en'):
    text = text.replace("_", " ")
    translated_text = translator.translate(text, dest=lang).text
    engine = pyttsx3.init()
    engine.say(translated_text)
    engine.runAndWait()

def predict_disease(image_path, lang='en'):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_probability = predictions[0][predicted_class_index]
    
    confidence_threshold = 0.9
    print(predicted_probability)
    if predicted_probability < confidence_threshold:
        return translator.translate("The uploaded image is not recognized as a tomato leaf.", dest=lang).text
    
    predicted_label = class_labels[predicted_class_index]
    # speak_text(f"The predicted disease is {predicted_label}", lang)
    return translator.translate(predicted_label.replace("_", " "), dest=lang).text

@app.route('/')
def index():
    file_count = count_uploaded_files()  # Count files dynamically
    return render_template('index.html', upload_count=file_count)

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    lang = request.form.get("language", "en")
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_disease = predict_disease(filepath, lang)
        print(predicted_disease)
        disease_info = {
  "Tomato   Leaf Mold": {
    "cause": "Caused by the fungus Passalora fulva, which thrives in humid conditions.",
    "pesticide": "Mancozeb or Chlorothalonil-based fungicides."
  },
  "Tomato   Early blight": {
    "cause": "Caused by the fungus Alternaria solani, which spreads through infected soil and plant debris.",
    "pesticide": "Copper-based fungicides or Chlorothalonil."
  },
  "Tomato   Bacterial spot": {
    "cause": "Caused by Xanthomonas bacteria, which spread through water splashes and infected seeds.",
    "pesticide": "Copper-based sprays or Streptomycin sulfate."
  },
  "Tomato   Tomato Yellow Leaf Curl Virus": {
    "cause": "Caused by Tomato yellow leaf curl virus (TYLCV), transmitted by whiteflies.",
    "pesticide": "Use insecticides like Imidacloprid to control whiteflies; no direct pesticide for the virus."
  },
  "Tomato   healthy": {
    "cause": "No disease detected, indicating a healthy plant.",
    "pesticide": "No pesticide required, but preventive sprays like Neem oil can be used for protection."
  }
}

        disease_details = disease_info.get(predicted_disease, {"cause": "Unknown", "pesticide": "No recommendation"})
        print(disease_details)
        file_count = count_uploaded_files()
        return render_template('index.html', filename=filename, predicted_disease=predicted_disease, language=lang, cause=disease_details["cause"],
            pesticide=disease_details["pesticide"],upload_count=file_count)

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=True)

