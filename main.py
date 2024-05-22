import numpy as np
import pandas as pd
import streamlit as st
from utils import *


# for custom CSS style
with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


st.header("Web-app-text-classification")

print("my file")
print("test2")