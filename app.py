import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Function to preprocess and predict the uploaded image
def predict_uploaded_image(uploaded_image):
    # Load the trained model
    model = load_model("potatoes.h5")

    # Define labels mapping
    labels = {0:'Potato_Early_blight', 1:'Potato_healthy',2: 'Potato_healthy'}

    # Preprocess and predict
    img_array = resize_and_preprocess_image(uploaded_image)
    prediction = model.predict(img_array)
    print(uploaded_image)
    print(prediction)
    print(np.argmax(prediction))
    predicted_class = labels[np.argmax(prediction)]
    print(predicted_class)
    

    return predicted_class

# Function to resize and preprocess an image
def resize_and_preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("plant disease prediction")
st.subheader("By:\n1.Manis Tyagi\n\n2.Pramod Arora\n\n3.Pratham Bhatia")
st.subheader("Under the guidance of: Dr. Shalu\n\nDepartment of Computer Science and Engineering")
st.write("\n")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
    # Make predictions and display result
    if st.button("Predict"):
        prediction = predict_uploaded_image(uploaded_image)
        st.success(f"Predicted Class: {prediction}")
