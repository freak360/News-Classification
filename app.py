import streamlit as st
import spacy
import gensim.downloader as api
import numpy as np
import pickle

# Load necessary libraries and models
nlp = spacy.load("en_core_web_lg")
wv = api.load("word2vec-google-news-300")
model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess text and return the vectorized form of text
def preprocess(text):
    doc = nlp(text)
    
    filtered_tokens = []
    for word in doc:
        if word.is_punct or word.is_stop:
            continue
        filtered_tokens.append(word.lemma_)
    return wv.get_mean_vector(filtered_tokens)



# Function to predict the result
def outcome(text):
    preprocessed_text = preprocess(text)
    stacked_text = np.stack([preprocessed_text])
    result = model.predict(stacked_text)
    if result == 0:
        print("Business")
    elif result == 1:
        print("Entertainment")
    elif result == 2:
        print("Politics")
    elif result == 3:
        print("Sport")
    else:
        print("Tech")

# Streamlit interface
st.title('Text Classification App')
user_input = st.text_area("Enter text here:", "Type here...")
if st.button('Predict'):
    category = outcome(user_input)
    st.write(f"The predicted category is: **{category}**")
