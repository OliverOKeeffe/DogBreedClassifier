import tensorflow as tf
import numpy as np
from PIL import Image

# If you trained EfficientNet with preprocess_input, set USE_EFFICIENTNET_PREPROCESS = True
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# If you trained with only rescale=1./255 (most common in your notebooks), leave this False.
# If you trained with tf.keras.applications.efficientnet.preprocess_input, set to True.
USE_EFFICIENTNET_PREPROCESS = False

# LOAD MODEL
MODEL_PATH = "saved_models/best_model_2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# CLASS NAMES (in the same order as training)
class_names = [
    "Afghan",
    "African Wild Dog",
    "Airedale",
    "American Hairless",
    "American Spaniel",
    "Basenji",
    "Basset",
    "Beagle",
    "Bearded Collie",
    "Bermaise",
    "Bichon Frise",
    "Blenheim",
    "Bloodhound",
    "Bluetick",
    "Border Collie",
    "Borzoi",
    "Boston Terrier",
    "Boxer",
    "Bull Mastiff",
    "Bull Terrier",
    "Bulldog",
    "Cairn",
    "Chihuahua",
    "Chinese Crested",
    "Chow",
    "Clumber",
    "Cockapoo",
    "Cocker",
    "Collie",
    "Corgi",
    "Coyote",
    "Dalmation",
    "Dhole",
    "Dingo",
    "Doberman",
    "Elk Hound",
    "French Bulldog",
    "German Sheperd",
    "Golden Retriever",
    "Great Dane",
    "Great Perenees",
    "Greyhound",
    "Groenendael",
    "Irish Spaniel",
    "Irish Wolfhound",
    "Japanese Spaniel",
    "Komondor",
    "Labradoodle",
    "Labrador",
    "Lhasa",
    "Malinois",
    "Maltese",
    "Mex Hairless",
    "Newfoundland",
    "Pekinese",
    "Pit Bull",
    "Pomeranian",
    "Poodle",
    "Pug",
    "Rhodesian",
    "Rottweiler",
    "Saint Bernard",
    "Schnauzer",
    "Scotch Terrier",
    "Shar_Pei",
    "Shiba Inu",
    "Shih-Tzu",
    "Siberian Husky",
    "Vizsla",
    "Yorkie"
]

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img)
    if USE_EFFICIENTNET_PREPROCESS:
        arr = effnet_preprocess(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predict_breed(image_path, top_k=5):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    probs = tf.nn.softmax(preds[0]).numpy()

    top_idx = probs.argsort()[-top_k:][::-1]
    top = [(class_names[i], float(probs[i])) for i in top_idx]

    # primary prediction
    breed, confidence = top[0]
    return breed, confidence, top


if __name__ == "__main__":
    test_image = "test_image.jpeg"  # update to a real image path
    breed, confidence, top = predict_breed(test_image, top_k=5)

    print(f"Predicted Breed: {breed}")
    print(f"Confidence: {confidence:.2%}")
    print("Top-5:")
    for name, p in top:
        print(f"  {name}: {p:.2%}")
