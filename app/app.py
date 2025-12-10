import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")

st.title("🐶 Dog Breed Classifier")
st.write("Upload an image of a dog and the model will predict its breed.")

# LOAD MODEL
MODEL_PATH = "saved_models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# CLASS NAMES (match training order)
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

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    processed = preprocess(img)
    preds = model.predict(processed)

    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    breed = class_names[class_idx]

    st.subheader("Prediction")
    st.write(f"**Breed:** {breed}")
    st.write(f"**Confidence:** {confidence:.2f}")
