import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os


# CONFIGURATION
# Path to your training dataset folder (used to auto-detect class names)
TRAIN_DIR = "Dataset/data/train"  # Change if your folder is elsewhere

# The input image size expected by the model
IMG_SIZE = (224, 224)

# Dictionary mapping model names (displayed in dropdown) to their file paths
MODEL_FILES = {
    "CNN (Custom)": "fish_cnn_model.h5",
    "MobileNetV2": "fish_mobilenet_model.h5"
}


# FUNCTION: Auto-extract class names from train folder
def get_class_names(train_dir):
    """
    Reads the subfolder names from the training directory.
    Each subfolder name corresponds to one fish species (class).
    
    Example:
        Dataset/data/train/Salmon
        Dataset/data/train/Tuna
        ...
    
    Returns:
        A sorted list of class names.
    """
    try:
        # Find all subfolders inside the training directory
        classes = sorted([
            folder for folder in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, folder))
        ])
        
        # Warn if no subfolders found
        if not classes:
            st.warning("No class folders found in the train directory!")
        return classes
    except Exception as e:
        # Show an error message if something goes wrong
        st.error(f"Error loading class names: {e}")
        return []

# Load class names automatically from the training folder
CLASS_NAMES = get_class_names(TRAIN_DIR)


# FUNCTION: Load a model from file (cached)
@st.cache_resource
def load_model(model_path):
    """
    Loads a saved TensorFlow/Keras model from disk.
    Uses Streamlit's cache_resource so the model loads only once
    (avoids reloading on every interaction).
    
    Args:
        model_path (str): Path to the .h5 model file.
        
    Returns:
        The loaded model object.
    """
    return tf.keras.models.load_model(model_path)


# STREAMLIT PAGE LAYOUT
st.set_page_config(page_title="Fish Image Classifier", layout="centered")
st.title("Fish Image Classifier")

# If no class names were loaded, stop the app
if not CLASS_NAMES:
    st.stop()

# Sidebar: Dropdown to choose which model to use
model_choice = st.sidebar.selectbox("Choose Model", list(MODEL_FILES.keys()))

# Load the chosen model
model = load_model(MODEL_FILES[model_choice])


# IMAGE UPLOAD
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])


# FUNCTION: Preprocess the uploaded image
def preprocess_image(image):
    """
    Prepares the uploaded image for prediction:
    - Resizes to the model's expected size
    - Ensures 3 channels (RGB)
    - Normalizes pixel values to [0, 1]
    - Expands dimensions to match model's input shape
    
    Args:
        image (PIL.Image): Uploaded image.
        
    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    # Resize the image
    img = image.resize(IMG_SIZE)
    x = np.array(img)

    # Handle grayscale image (shape = (H, W))
    if len(x.shape) == 2:
        x = np.stack([x] * 3, axis=-1)
    # Handle RGBA image (shape = (H, W, 4)) → remove alpha channel
    elif x.shape[2] == 4:
        x = x[..., :3]

    # Normalize pixel values
    x = x / 255.0
    # Add batch dimension (1, height, width, channels)
    x = np.expand_dims(x, axis=0)
    return x


# PREDICTION AND OUTPUT
if uploaded_file:
    # Open uploaded image
    image = Image.open(uploaded_file)
    
    # Display uploaded image in the app
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess for model input
    x = preprocess_image(image)
    
    # Make prediction
    preds = model.predict(x)
    
    # Get predicted class
    pred_class = CLASS_NAMES[np.argmax(preds)]
    
    # Get prediction confidence (highest probability)
    confidence = np.max(preds) * 100

    # Display prediction result
    st.success(f"**Prediction:** {pred_class} ({confidence:.2f}%)")

    # Show all class probabilities in a bar chart
    st.subheader("Class Probabilities:")
    prob_dict = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_dict)


# SIDEBAR INFO
st.sidebar.markdown("---")
st.sidebar.info(
    "Upload a clear fish image. "
    "Model and class names are automatically detected based on your data folder."
)
