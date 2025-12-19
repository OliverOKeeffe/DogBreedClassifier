# Dog Breed Classifier

A deep learning project that classifies dog breeds using a Convolutional Neural Network (CNN) with transfer learning. The model is trained on 70 different dog breeds and deployed as an interactive web application.

## Features

- **CNN Models**: MobileNetV2 and EfficientNetB0 available in the Streamlit app
- **70 Dog Breeds**: Classified with high accuracy
- **Web Interface**: Easy-to-use Streamlit application with model selection and Top-K control
- **Model Checkpointing**: Saves the best model during training
- **Early Stopping**: Prevents overfitting

## Project Structure

```
DogBreedClassifier/
├── app/
│   └── app.py                 # Streamlit web application
├── data/
│   └── AI-CA-Data/
│       ├── train/             # Training images (70 breed folders)
│       ├── valid/             # Validation images
│       └── test/              # Test images
├── notebooks/
│   ├── train_cnn_MobileNet.ipynb                 # MobileNetV2 training
│   └── train_cnn_EfficientNetB0.ipynb            # EfficientNetB0 training
├── saved_models/
│   ├── best_model.h5                              # MobileNetV2 weights
│   └── efficientnet_best_model.h5                 # EfficientNetB0 weights
├── src/
│   ├── MobileNet_predict.py                       # MobileNetV2 prediction script
│   └── EfficentNetB0_predict.py                   # EfficientNetB0 prediction script
├── requirements.txt           # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB+ free disk space (for model and data)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/OliverOKeeffe/DogBreedClassifier.git
cd DogBreedClassifier
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Pillow (PIL)
- Streamlit

## Running the Project

### Option 1: Web Application (Recommended)

```bash
streamlit run app/app.py
```

The web app will open in your browser at `http://localhost:8501`. Use the sidebar to select **MobileNetV2** or **EfficientNetB0** and choose the Top-K output.

### Option 2: Use the Prediction Script

Pick the script for your model:

- MobileNetV2:
  ```bash
  python src/MobileNet_predict.py path/to/image.jpg
  ```
- EfficientNetB0:
  ```bash
  python src/EfficentNetB0_predict.py path/to/image.jpg
  ```

Adjust the `--model` argument if you saved checkpoints under a different name.

### Option 3: Train the Model (Advanced)

Open the Jupyter notebooks:

```bash
jupyter notebook notebooks/train_cnn_MobileNet.ipynb
```

Or for EfficientNetB0:

```bash
jupyter notebook notebooks/train_cnn_EfficientNetB0.ipynb
```

Follow the cells to:

1. Load and preprocess data
2. Build the CNN model
3. Train the model
4. Evaluate on test data

## Usage

1. **Web App**: Upload an image of a dog, and the model will predict the breed and confidence score
2. **Script**: Modify `predict.py` to point to your image path and run it
3. **Notebook**: Train your own model with different hyperparameters

## Model Details

- **Base Models**: MobileNetV2 and EfficientNetB0 (pre-trained on ImageNet)
- **Input Size**: 224×224 pixels
- **Output**: 70 dog breed classes
- **Training Parameters**:
  - Epochs: 10-15
  - Batch Size: 32
  - Optimizer: Adam
  - Loss: Categorical Cross-entropy

## Dataset

The model is trained on the AI-CA-Data dataset with 70 dog breed categories:

- Afghan, Beagle, Bulldog, Chihuahua, Golden Retriever, Labrador, Poodle, Rottweiler, Siberian Husky, and more...

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'tensorflow'`

- Solution: Reinstall dependencies with `pip install -r requirements.txt`

**Issue**: Out of memory errors during training

- Solution: Reduce `batch_size` in the notebook or use GPU acceleration

**Issue**: Model file not found

- Solution: Ensure `best_model.h5` exists in `saved_models/` directory

## Performance

- Validation Accuracy: ~85-90%
- Training Time: ~10-15 minutes (on CPU)

## Future Improvements

- Fine-tune the model with more epochs
- Add confidence threshold filtering
- Support for multiple dog detection
- Mobile app deployment

## License

This project is open source and available for educational purposes.

## Author

Oliver O'Keeffe

---

**Questions?** Feel free to open an issue on GitHub!
