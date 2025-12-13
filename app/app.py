import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")

# -------------------------
# Styling: Colors & Fonts
# -------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Jacques+Francois&display=swap');
      
      /* Force Jacques Francois on ALL text */
      * { font-family: 'Jacques Francois', serif !important; }
      
            /* Page background - white */
            .stApp { 
                background: #ffffff; 
                padding: 0;
                padding-top: 5.25rem; /* reserve space for fixed header */
                font-family: 'Jacques Francois', serif !important;
                color: #333333;
            }
      
            /* Green header bar - fixed to top */
            .header-bar {
                background: #90EE90;
                padding: 0.75rem 1.5rem;
                text-align: center;
                margin: 0;
                border-radius: 0;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 9999;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 5rem; /* consistent header height */
            }
      
            /* Title in header */
            h1 { 
                color: #1a1a1a; 
                font-family: 'Jacques Francois', serif !important; 
                background: #90EE90; /* restore green background */
                padding: 0.5rem 1rem;
                margin: 0;
                text-align: center;
                line-height: 1;
                font-size: 1.75rem;
                border-radius: 4px;
            }
      
      /* Headings */
      h2 { color: #333333; font-family: 'Jacques Francois', serif !important; }
      h3 { color: #333333; font-family: 'Jacques Francois', serif !important; }
      
      /* Text color */
      p { color: #333333; font-family: 'Jacques Francois', serif !important; }
      
      /* Markdown text */
      .stMarkdown { font-family: 'Jacques Francois', serif !important; color: #333333; }
      
      /* Buttons */
      .stButton>button { 
        background: #90EE90; 
        color: #000;
        border: none; 
        padding: 8px 14px; 
        border-radius: 6px;
        font-family: 'Jacques Francois', serif !important;
        font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐶 Dog Breed Classifier")
st.markdown("### Upload an image of a dog and the model will predict its breed.")

# Single model configuration (reverted to original behavior)
DEFAULT_MODEL_PATH = "saved_models/best_model.h5"


@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

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


def preprocess(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    # Use LANCZOS resampling; Pillow 10+ uses Image.Resampling
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.LANCZOS
    img = ImageOps.fit(img, target_size, method=resample)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_top_k(preprocessed_image, model, class_names, top_k=3):
    preds = model.predict(preprocessed_image)
    probs = tf.nn.softmax(preds[0]).numpy()
    top_idx = probs.argsort()[-top_k:][::-1]
    return [(class_names[i], float(probs[i])) for i in top_idx]


# UI: Top-K selector in the sidebar (fixed single model)
model = load_model(DEFAULT_MODEL_PATH)

top_k = st.sidebar.slider("Top Dog Breeds", min_value=1, max_value=10, value=3)

st.markdown("#### Choose a dog image...")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"]) 

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    processed = preprocess(img)

    with st.spinner("Predicting..."):
        results = predict_top_k(processed, model, class_names, top_k=top_k)

    st.markdown("### Top Predictions")

    # Extract names and raw probabilities
    names = [r[0] for r in results]
    probs = [r[1] for r in results]

    # If top_k == 1, show raw confidence; otherwise normalize so top-k sum to 100%
    if top_k == 1:
        # Show raw model confidence (not normalized)
        for name, prob in zip(names, probs):
            st.markdown(f"### - **{name}** — {prob:.2%}")
        norm_probs = probs  # use raw for chart too
    else:
        # Normalize so the shown percentages for the selected Top-K sum to 100%
        total = sum(probs)
        if total > 0:
            norm_probs = [p / total for p in probs]
        else:
            norm_probs = [0 for _ in probs]
        # Display normalized percentages (and show original absolute prob in parens)
        for name, nprob, aprob in zip(names, norm_probs, probs):
            st.markdown(f"### - **{name}** — {nprob:.2%} (raw: {aprob:.2%})")

    # show a bar chart of the normalized Top-K probabilities
    try:
        import pandas as pd

        df = pd.DataFrame({"breed": names, "probability": norm_probs}).set_index("breed")
        st.bar_chart(df)
    except Exception:
        st.write({n: f"{p:.2%}" for n, p in zip(names, norm_probs)})
