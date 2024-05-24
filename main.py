import numpy as np
import pandas as pd
import joblib,os
import spacy
nlp = spacy.load("en_core_web_sm")
import streamlit as st
import matplotlib.pyplot as plt 
import matplotlib
from utils import CnnClassifier, get_category,get_cat, LrClassifier
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

# for custom CSS style
# with open("style.css") as f:
#     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    


activity = ['Machine learning','Deep learning', 'NLP process']
st.sidebar.title("NLP")
st.sidebar.markdown("""
Le traitement du langage naturel(NLP) consiste à utiliser des ordinateurs pour traiter du texte ou de la parole en langage naturel. 
""")
with st.sidebar:
    choice = st.radio(
        "Selectionner un type de model",activity)


html_temp = """
<div style="background-color:blue;padding:10px">
<h1 style="color:white;text-align:center;">Web-app-text-classification </h1>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
# image = Image.open('images/speech-text.png')
image = Image.open('img/nlp.jpeg')
st.image(image)


if choice == 'Machine learning':
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
                with st.spinner("Votre texte est en cours de prédiction..."):
                    if "Logistic Regression" in options and len(options) == 1:

                        st.text("Original Text::\n{}".format(text))
                        model = LrClassifier()
                        # text_vector = model.read_extract_text_file(text)
                        # text_vector = model.convert_text_to_vector_lr(text)
                        text_vector = model.predict_result(text)
                        predictions, pred_proba = model.predict(text_vector)
                    
                    else:
                        st.error(
                            "Veuillez sélectionner un modèle à prédire")
                        st.stop()

                col1, col2 = st.columns(2)
                with col1:
                    predicted_category = get_category(pred_proba)
                    st.write("Catégorie prédit")
                    st.success(predicted_category)

                with col2:
                    st.write("Score de prédiction:")
                    st.success(f"{np.max(pred_proba)*100:.2f} %")

                st.write("Graphique de prédiction:")
                df = pd.DataFrame(data=pred_proba, columns=[
                                  "Business", "Entertainment", "Politics", "Sports", "Technology"])
                st.bar_chart(df.T, height=550)
            else:
                st.error(
                'Veuillez saisir un texte qui souhaite être prédit')


if choice == 'Deep learning':
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
                with st.spinner("Votre texte est en cours de prédiction..."):
                    if "CNN" in options and len(options) == 1:
                        model = CnnClassifier()
                        text_vector = model.convert_text_to_vector(text)
                        predictions = model.predict(text_vector)
    
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