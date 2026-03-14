#We add below import steamlit in order to use the steamlit function 
import streamlit as st

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import numpy as np

# Testing with the saved model
model2 = AutoModelForSequenceClassification.from_pretrained("CustomModel_yelp",
                                                            num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenized testing data
label = 4 # label = 4
text = "dr. goldberg offers everything i look for in a general practitioner. he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first. really, what more do you need? i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank."
inputs = tokenizer(text,
                   padding=True,
                   truncation=True,
                   return_tensors='pt')

outputs = model2(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()

# Get the index of the largest output value
max_index = np.argmax(predictions)

print(f"The label is {label} and the predicted label is {max_index}")
