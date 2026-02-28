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
age_classifier = pipeline("image-classification",
                          model="nateraw/vit-age-classifier")

image_name = "middleagedMan.jpg"
image_name = Image.open(image_name).convert("RGB")

# Classify age
age_predictions = age_classifier(image_name)
print(age_predictions)
age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

# Display results
print("Predicted Age Range:")
print(f"Age range: {age_predictions[0]['label']}")
#Since this is a steamlit application, therefore cannot just use 'Print' but using a steamlit function instead to show
st.write("Predicted Age Range:")
st.write(f"Age range: {age_predictions[0]['label']}")


#We add below function of streamlit to check if the programe is Done or not
st.write("Done")
