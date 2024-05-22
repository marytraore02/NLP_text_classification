import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.utils import pad_sequences


idx2category = {
  0: "Business",
  1: "Entertainment",
  2: "Politics",
  3: "Sports",
  4: "Technology"
}

print('mary')