import numpy as np
import pandas as pd
import streamlit as st
from utils import CnnClassifier, get_category


# for custom CSS style
with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


st.header("Web-app-text-classification")

with st.form(key="input_form"):
    options = st.multiselect(
        'Selectionner un model',
        ['CNN'])

    text = st.text_area(label="Text d'entrer", height=200,
    placeholder="Entrez le texte")
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        if text != "":
            with st.spinner("Your text is being predicted..."):
                if "CNN" in options and len(options) == 1:
                    model = CnnClassifier()
                    text_vector = model.convert_text_to_vector(text)
                    predictions = model.predict(text_vector)

                # elif "GRU" in options and len(options) == 1:
                #     model = GruClassifier()
                #     text_vector = model.convert_text_to_vector(text)
                #     predictions = model.predict(text_vector)

                # elif len(options) == 2:
                #     lstm_model = LstmClassifier()
                #     lstm_text_vector = lstm_model.convert_text_to_vector(text)
                #     lstm_predictions = lstm_model.predict(lstm_text_vector)

                #     gru_model = GruClassifier()
                #     gru_text_vector = gru_model.convert_text_to_vector(text)
                #     gru_predictions = gru_model.predict(gru_text_vector)

                #     predictions = (lstm_predictions + gru_predictions) / 2

                else:
                    st.error(
                        "Veuillez sélectionner un modèle à prédire")
                    st.stop()

            col1, col2 = st.columns(2)
            with col1:
                predicted_category = get_category(predictions)
                st.write("Catégorie prédit")
                st.success(predicted_category)

            with col2:
                st.write("Score de prédiction:")
                st.success(f"{np.max(predictions)*100:.2f} %")

            st.write("Graphique de prédiction:")
            df = pd.DataFrame(data=predictions, columns=[
                              "Business", "Entertainment", "Politics", "Sports", "Technology"])
            st.bar_chart(df.T, height=550)
        else:
            st.error(
                'Veuillez saisir un texte qui souhaite être prédit')

