import pickle
import numpy as np
import pandas as pd
import joblib,os
from tensorflow import keras
from keras.utils import pad_sequences

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
        text = pd.Series(text)
        text_vector = self.cnn_tokenizer.texts_to_sequences(text)
        text_vector_padded = pad_sequences(text_vector,
                                           maxlen=500,
                                           padding="post",
                                           truncating="post")

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
    
    def predict_result(self, text: str):
        text = [f'{text}']
        print(text)
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





    

