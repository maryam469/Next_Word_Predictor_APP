import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model('model_word_lstm.h5')  # apna trained model ka path do
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_sequence_len = 14  # jo aapne training time pe use kiya tha

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = predicted.argmax(axis=-1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# Streamlit UI
st.title("Next Word Predictor")

user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
        st.success(f"Predicted next word: **{next_word}**")
    else:
        st.warning("Please enter some text.")
