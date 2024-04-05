from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('catsvdogsv1.h5')

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize the image to match the input size of the model
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def main():
    st.title("Image Classification App")
    st.write("Upload an image for classification")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image and make prediction
        img = preprocess_image(uploaded_file)
        prediction = loaded_model.predict(img)
        class_names = ['cat', 'dog']  # Modify according to your class labels
        result = class_names[np.argmax(prediction)]

        st.write("Prediction:", result)

if __name__ == '__main__':
    main()


