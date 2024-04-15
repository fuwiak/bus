import streamlit as st
import requests
from PIL import Image
import io

st.title('People Count Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Prepare the file to send to the FastAPI
    files = {'file': uploaded_file.getvalue()}

    # Post the file to the FastAPI endpoint for prediction
    response = requests.post('http://localhost:8000/predict/', files=files)

    if response.status_code == 200:
        response_data = response.json()
        predicted_count = response_data['predicted_count']
        print(predicted_count)

        # Display the predicted number of people
        st.write(f"Predicted number of people: {predicted_count}")
    else:
        print(response.text)
        print(response.status_code)

        st.error("Failed to process the image.")
