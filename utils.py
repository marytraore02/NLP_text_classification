import pickle
import numpy as np
import pandas as pd
import joblib, os
import re
import unidecode
import string
string.punctuation
import streamlit as st
import time
from tensorflow import keras
from keras.utils import pad_sequences
import spacy
nlp = spacy.load("en_core_web_sm")
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Télécharge les stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Charge l'ensemble des mots vide en anglais
stopwords_en = set(stopwords.words('english'))
# st.text(stopwords_en)

# Déclaration d'un dictionnaire qui prend les catégories et leurs index
dict_category = {
    0: "Business",
    1: "Entertainment",
    2: "Politics",
    3: "Sports",
    4: "Technology"
}

def get_cat(predictions: np.ndarray) -> str:
    predicted_index = dict_category[predictions[0]]
    print(predicted_index)
    return predicted_index

def get_category(predictions: np.ndarray) -> str:
    """
  Get the category of the maximum score in `predictions`.

  Parameters:
  ------------
  - predictions : np.ndarray
      Contains 5 scores repesenting the predcited score for each category.

  Returns:
  --------
  - str
      Represents the predicted score using `dict_category` dictionary.
  """
    predicted_index = np.argmax(predictions)
    return dict_category[predicted_index]

def progress():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
        if percent_complete == 50:
            st.write("**Graphique de prédiction:**")
        my_bar.progress(percent_complete + 1)

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def stem_words(tokens):
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(word) for word in tokens if word not in stopwords_en]
    return stems
    
def lemmatize_words(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in tokens if word not in stopwords_en]
    return lemmas

def preprocess_data(text):
    text = remove_html_tags(text)
    # Supprimer la ponctuation
    text_no_punctuation = "".join([ch for ch in text if ch not in string.punctuation])

    # Remplacer les sauts de ligne et tabulations par espace
    text_no_newline = text_no_punctuation.replace("\n", " ").replace("\t"," ")

    # Convertir les caractères accentués en leurs équivalents non accentués
    text_no_accent = unidecode.unidecode(text_no_newline)

    # Supprimer les chiffres
    text_no_number = re.sub(r'\d+', '', text_no_accent)

    # Rendre les textes miniscule
    text_lower = text_no_number.lower()

    # Remplacer les doubles espaces par un espace
    text_no_space = " ".join(text_lower.split())

    # Tokeniser les textes
    tokens = re.split('\W+', text_no_space)

    text = stem_words(tokens)
    # text = lemmatize_words(tokens)
    cleaned_text = ' '.join(text)
    
    return cleaned_text



class CnnClassifier:
    """Responsible for make predictions on a text using LSTM-based model"""

    def __init__(self) -> None:
        self.cnn_model = keras.models.load_model("models/cnn_model.h5")
        with open("tokenizers/tokenizer.pickle", 'rb') as handle:
            self.cnn_tokenizer = pickle.load(handle)

    def convert_text_to_vector(self, text: str) -> np.ndarray:
        """
      Convert passed `text` to a numerical vector.

      Parameters:
      ------------
      - text : str
          Text needed to be converted to vector.

      Returns:
      --------
      - text_vector_padded : np.ndarray
          A vector representation of size 256 of `text`.
      """
        # st.write("Original Text::\n{}".format(text))
        text = preprocess_data(text)
        text = pd.Series([f'{text}'])
        # st.write(text)
        st.text("processed text::\n{}".format(text))
        # print(len(text))
        # print(type(text))
        text_vector = self.cnn_tokenizer.texts_to_sequences(text)
        # st.text("vector text::\n{}".format(text_vector))
        
        text_vector_padded = pad_sequences(text_vector,
                                           maxlen=500,
                                           padding="post",
                                           truncating="post")

        # st.text("text_vector_padded::\n{}".format(text_vector_padded))

        return text_vector_padded

    def predict(self, text_vector: np.ndarray) -> np.ndarray:
        """
      Get predictions scores of the passes `text_vector`

      Parameters:
      ------------
      - text_vector : np.ndarray
          A vector representation of a text of size 256

      Returns:
      --------
      - predictions : np.ndarray
          A vector of size number of categories "5" represents
          the score of each category.
      """
        predictions = self.cnn_model.predict(text_vector)
        print(predictions)
        return predictions


class LrClassifier:
    """ test"""

    def __init__(self) -> None:
        with open("models/lr_model.pkl", 'rb') as handle:
            self.lr_model = pickle.load(handle)
        with open("tokenizers/vectorizer.pkl", 'rb') as handle:
            self.lr_vectorizer = pickle.load(handle)

    def transform_text(self, text: str):
        # text = remove_html_tags(text)
        # print(type(text))
        text = preprocess_data(text)
        print(type(text))
        text = [f'{text}']
        # text = pd.Series([f'{text}'])
        st.text("processed text::\n{}".format(text))
        # st.write(text)
        # print(text)
        text_vector = self.lr_vectorizer.transform(text).toarray()
        print(text_vector)
        print(type(text_vector))
        return text_vector

    def predict(self, text_vector: np.ndarray) -> np.ndarray:
        prediction = self.lr_model.predict(text_vector)
        pred_proba = self.lr_model.predict_proba(text_vector)
        print(prediction)
        print(pred_proba)
        print(type(prediction))
        pred = dict_category[prediction[0]]
        print(pred)
        return prediction, pred_proba
