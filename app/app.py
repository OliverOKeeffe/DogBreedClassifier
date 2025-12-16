import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="Dog Breed Classifier", 
    page_icon="🐶", 
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None
)

# -------------------------
# Styling: Colors & Fonts
# -------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Jacques+Francois&display=swap');
      
      /* Hide Streamlit toolbar completely */
      header[data-testid="stHeader"] {
          display: none !important;
          visibility: hidden !important;
          height: 0 !important;
      }
      
      .stApp > header {
          display: none !important;
      }
      
      /* Force Jacques Francois on ALL text */
      * { font-family: 'Jacques Francois', serif !important; }
      
      /* Page background - white */
      .stApp { 
          background: #ffffff; 
          padding: 0;
          padding-top: 0;
          font-family: 'Jacques Francois', serif !important;
          color: #333333;
      }
      
      /* Remove default Streamlit container padding */
      .block-container {
          padding-top: 8rem !important;
          padding-left: 2rem !important;
          padding-right: 2rem !important;
      }
      
      /* Title styling - full-width green header bar using position fixed */
      h1 { 
          color: #1a1a1a; 
          font-family: 'Jacques Francois', serif !important; 
          background: #90EE90;
          padding: 1.5rem 2rem;
          margin: 0;
          text-align: center;
          line-height: 1.4;
          font-size: 2.2rem;
          border-radius: 0;
          position: fixed;
          top: 0;
          left: 7rem;
          right: 0;
          width: 100%;
          z-index: 999;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      
      /* Headings */
      h2 { color: #333333; font-family: 'Jacques Francois', serif !important; margin-top: 1.5rem; }
      h3 { color: #333333; font-family: 'Jacques Francois', serif !important; margin-top: 1rem; }
      
      /* Text color */
      p { color: #333333; font-family: 'Jacques Francois', serif !important; }
      
      /* Markdown text */
      .stMarkdown { font-family: 'Jacques Francois', serif !important; color: #333333; }
      
      /* Sidebar styling */
      section[data-testid="stSidebar"] {
          background: #f8f9fa;
          padding-top: 2rem;
      }
      
      section[data-testid="stSidebar"] > div {
          padding-top: 2rem;
      }
      
      /* Hide sidebar collapse button to keep sidebar always visible */
      button[kind="header"] {
          display: none !important;
      }
      
      button[kind="headerNoPadding"] {
          display: none !important;
      }
      
      [data-testid="collapsedControl"] {
          display: none !important;
      }
      
      [data-testid="stSidebarCollapseButton"] {
          display: none !important;
      }
      
      section[data-testid="stSidebar"] button[aria-label*="Close"] {
          display: none !important;
      }
      
      /* Buttons */
      .stButton>button { 
        background: #90EE90; 
        color: #000;
        border: none; 
        padding: 10px 20px; 
        border-radius: 6px;
        font-family: 'Jacques Francois', serif !important;
        font-weight: 600;
      }
      
      /* File uploader */
      .stFileUploader {
          margin: 1rem 0;
      }
      
      /* Images */
      img {
          border-radius: 8px;
          margin: 1rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐶 Dog Breed Classifier")
st.markdown("### Upload an image of a dog and the model will predict its breed.")

# Available models
MODELS = {
    "MobileNetV2": "saved_models/best_model.h5",
    "EfficientNetB0": "saved_models/efficientnet_best_model.h5"
}


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


def preprocess(img: Image.Image, model_name: str, target_size=(224, 224)):
    img = img.convert("RGB")
    # Use LANCZOS resampling; Pillow 10+ uses Image.Resampling
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.LANCZOS
    img = ImageOps.fit(img, target_size, method=resample)
    arr = np.array(img)
    
    # Apply model-specific preprocessing
    if model_name == "EfficientNetB0":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input # type: ignore
        arr = preprocess_input(arr)
    else:  # MobileNetV2 and default
        arr = arr / 255.0
    
    return np.expand_dims(arr, axis=0)


def predict_top_k(preprocessed_image, model, class_names, top_k=3):
    preds = model.predict(preprocessed_image)
    probs = tf.nn.softmax(preds[0]).numpy()
    top_idx = probs.argsort()[-top_k:][::-1]
    return [(class_names[i], float(probs[i])) for i in top_idx]


# UI: Model selector and Top-K selector in the sidebar
selected_model_name = st.sidebar.selectbox("Choose Model", list(MODELS.keys()))
model_path = MODELS[selected_model_name]
model = load_model(model_path)

top_k = st.sidebar.slider("Top Dog Breeds", min_value=1, max_value=10, value=3)

st.markdown("#### Choose a dog image...")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"]) 

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    processed = preprocess(img, selected_model_name)

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
