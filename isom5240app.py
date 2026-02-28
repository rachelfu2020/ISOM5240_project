from transformers import pipeline
from PIL import Image
#We add below import steamlit in order to use the steamlit function 
import streamlit as st


# Streamlit UI
print("Title: Age Classification using ViT")
#Since this is a steamlit application, therefore cannot just use 'Print' but using a steamlit function instead to show
st.header("Title: Age Classification using ViT")

# Load the age classification pipeline
# The code below should be placed in the main part of the program
age_classifier = pipeline("image-classification", model="prithivMLmods/Age-Classification-SigLIP2")


def main():
    st.set_page_config(page_title="Your Image to Age Classification", page_icon="🦜")
    st.header("Turn Your Image to Age")
    uploaded_file = st.file_uploader("Select an Image...")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
      
#image_name = "middleagedMan.jpg"
#image_name = Image.open(image_name).convert("RGB")

# Classify age
age_predictions = age_classifier(uploaded_file)
print(age_predictions)
#Since this is a steamlit application, therefore cannot just use 'Print' but using a steamlit function instead to show
st.write(age_predictions)
age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

# Display results
print("Predicted Age Range:")
print(f"Age range: {age_predictions[0]['label']}")
#Since this is a steamlit application, therefore cannot just use 'Print' but using a steamlit function instead to show
st.write("Predicted Age Range:")
st.write(f"Age range: {age_predictions[0]['label']}")


#We add below function of streamlit to check if the programe is Done or not
st.write("Done")
