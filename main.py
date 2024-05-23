import numpy as np
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import streamlit as st
import matplotlib.pyplot as plt 
import matplotlib
from utils import CnnClassifier, get_category, LrClassifier
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# for custom CSS style
# with open("style.css") as f:
#     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
st.set_option('deprecation.showPyplotGlobalUse', False)

# st.header("Web-app-text-classification")

# st.title("News Classifier")
# st.subheader("ML App with Streamlit")


activity = ['Machine learning','Deep learning', 'NLP process']
st.sidebar.subheader("Selectionner un type de model")
with st.sidebar:
    choice = st.radio(
        "Sélection",activity)


html_temp = """
<div style="background-color:blue;padding:10px">
<h1 style="color:white;text-align:center;">Web-app-text-classification </h1>
</div>

"""


if choice == 'Machine learning':
    st.markdown(html_temp,unsafe_allow_html=True)
    st.info("Machine Learning content")
    with st.form(key="input_form"):
        options = st.multiselect(
            'Selectionner un model',
            ['Logistic Regression'])

        text = st.text_area(label="Text d'entrer", height=200,
        placeholder="Entrez le texte")

        submitted = st.form_submit_button("Submit")
        if submitted:
            if text != "":
                with st.spinner("Your text is being predicted..."):
                    if "Logistic Regression" in options and len(options) == 1:
                        model = LrClassifier()
                        predictions = model.predict(text)

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


if choice == 'Deep learning':
    st.markdown(html_temp,unsafe_allow_html=True)
    st.info("Deep learning content")
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


if choice == 'NLP process':
    st.markdown(html_temp,unsafe_allow_html=True)
    st.info("Natural Language Processing of Text")
    raw_text = st.text_area("Entrez le texte")
    nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
    task_choice = st.selectbox("Choisissez la tâche PNL",nlp_task)
    if st.button("Analyze"):
        if raw_text != "":
            st.info("Original Text::\n{}".format(raw_text))
            docx = nlp(raw_text)
            if task_choice == 'Tokenization':
                result = [token.text for token in docx ]
            elif task_choice == 'Lemmatization':
                result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
            elif task_choice == 'NER':
                result = [(entity.text,entity.label_)for entity in docx.ents]
            elif task_choice == 'POS Tags':
                result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]
            st.json(result)
        else:
            st.error(
            'Veuillez saisir un texte qui sera traiter')

    

    if st.button("Tabuler"):
        if raw_text != "":
            docx = nlp(raw_text)
            c_tokens = [token.text for token in docx ]
            c_lemma = [token.lemma_ for token in docx ]
            c_pos = [token.pos_ for token in docx ]
        
            new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
            st.dataframe(new_df)
        else:
            st.error(
            'Veuillez saisir un texte qui sera traiter')

    
    if st.checkbox("WordCloud"):
        if raw_text != "":
            c_text = raw_text
            wordcloud = WordCloud().generate(c_text)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
        else:
            st.error(
            'Veuillez saisir un texte qui sera traiter')