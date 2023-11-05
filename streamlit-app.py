import streamlit as st
from PIL import Image
import numpy as np
import keras
import pickle


# Load the saved models for expression and gender prediction
expression_model = keras.models.load_model("ExModel")
gender_model = keras.models.load_model("GenModel")

with open('exp_pca.pkl', 'rb') as pickle_file:
    exp_pca = pickle.load(pickle_file)

with open('gen_pca.pkl', 'rb') as pickle_file:
    gen_pca = pickle.load(pickle_file)

# Function to preprocess the image for prediction
def preprocess_image(image):
    img = image.resize((100, 100))
    img = np.array(img)
    img2 = img/255.0
    img3 = img2.reshape(1, img.shape[0]*img.shape[1])
    exp_img = exp_pca.transform(img3)
    gen_img = gen_pca.transform(img3)
    return exp_img, gen_img

# Function to make predictions
def predict_image(image):
    # Preprocess the image
    exp_img, gen_img = preprocess_image(image)
    exp_img = exp_img.flatten()
    gen_img = gen_img.flatten()

    # Make predictions
    expression_prediction = expression_model.predict(np.expand_dims(exp_img, axis=0))
    gender_prediction = gender_model.predict(np.expand_dims(gen_img, axis=0))

    return expression_prediction, gender_prediction

st.title("Face Expression and Gender Recognition App")
st.sidebar.markdown("### Upload an Image")

# Create a button to upload an image
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image using OpenCV and convert it to grayscale
    # image_data = io.BytesIO(uploaded_file.read())
    # image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Read the uploaded image using Pillow
    image = Image.open(uploaded_file)

    # Convert the image to grayscale
    image_gray = image.convert("L")
    
    # Display the uploaded image in grayscale
    st.image(image_gray, caption="Uploaded Image (Grayscale)", use_column_width=True, channels="GRAY")

    # Make predictions
    expression_pred, gender_pred = predict_image(image_gray)

    # Define expression and gender labels
    expression_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    gender_labels = ["Female", "Male"]

    # Display the expression and gender predictions in a table
    st.markdown("### Predictions:")
    st.table({
        "Expression": expression_labels[np.argmax(expression_pred)],
        "Gender": gender_labels[np.argmax(gender_pred)],
    })
