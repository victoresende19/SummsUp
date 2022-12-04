# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:18:08 2022

@author: Victor Resende
"""
import streamlit as st
import pdfplumber
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import evaluate
from io import StringIO


def write_html(html: str):
    return st.markdown(html, unsafe_allow_html=True)


def file_upload(file):
    if file is not None:
        pdf = extract_data(file)
        return pdf


def extract_data(doc):
    pdf = pdfplumber.open(doc)
    text = ''
    for page in pdf.pages:
        text = text + page.extract_text()
        text = text + ' '
    return text


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def portuguese_sumarization(text):
    """'
    Sumariza o texto disponibilizado (em português)

    recebe - texto: texto disponibilizado 
    retorna - texto: texto sumarizado (em português)
    """

    token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
    model_name = 'phpaiola/ptt5-base-summ-xlsum'

    tokenizer = T5Tokenizer.from_pretrained(token_name)
    model_pt = T5ForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode(text, max_length=512,
                              truncation=True, return_tensors='pt')
    summary_ids = model_pt.generate(
        inputs, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0])

    return write_html(summary)


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def english_sumarization(text):
    """'
    Sumariza o texto disponibilizado (em inglês)

    recebe - texto: texto disponibilizado 
    retorna - texto: texto sumarizado (em inglês)
    """

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def acc_sumarization(texto: str, resumo: str) -> str:
    """'
    Retorna a acurácia do resumo por meio da métrica Harim.

    recebe - texto: texto disponibilizado, resumo: texto resumido pelo modelo
    retorna - acuracia: acuracia do resumo
    """

    HARIM = evaluate.load('NCSOFT/harim_plus')

    texto_cru = [texto]
    texto_resumido = [resumo]
    acuracia = HARIM.compute(references=texto_cru, predictions=texto_resumido)

    return round(acuracia[0], 4)


def display_sumarization(text, language):
    if language == 'Português':
        return portuguese_sumarization(text)
    elif language == 'Inglês':
        return english_sumarization(text)


st.set_page_config(
    page_icon='🎈', page_title='Sumarizador de textos')

write_html(
    "<h1 style='text-align: center; color: black;'> 📋 Sumarizador de textos 📋 </h1>")
expander = st.expander(label="🛈 Sobre o aplicativo", expanded=True)
expander.markdown(
    """
        - O *Sumarizador de Textos* é uma interface fácil de usar construída em Stramlit para criar para o resumo de textos digitados pelo usuário ou PDF.
        - O aplicativo utiliza redes neurais pré-treinadas que aproveitam várias incorporações de NLP e depende de [Transformers](https://huggingface.co/transformers/).
        - Além disso, a aplicação conta com suporte para resumir dois tipos de idiomas: Português e Inglês! 🤗 
        - Para mais informações ou sugestões, contate o autor: [Victor Resende](https://www.linkedin.com/in/victor-resende-508b75196/). 
    """
)
st.markdown("")
st.markdown("")


text_type = st.selectbox('Que maneira gostaria de resumir seu texto?',
                         ('Escolha as opções', 'Resumo escrito', 'Resumo em PDF'))

if text_type == 'Resumo escrito':
    form = st.form(key='my_form')
    language = form.selectbox('Qual a língua do PDF?', ('Português', 'Inglês'))
    text = form.text_area(
        "Texto a ser resumido:",
        height=510,
        placeholder='Escreva aqui...'
    )
    submit_button = form.form_submit_button(label='✨ Resumir!')

    if submit_button:
        with st.spinner('Resumindo...'):
            st.markdown(f'{display_sumarization(text, language)}',
                        unsafe_allow_html=True)
            #acc_sumarization(text, display_sumarization(text, language))

elif text_type == 'Resumo em PDF':
    form = st.form(key='my_form')
    language = form.selectbox('Qual a língua do PDF?', ('Português', 'Inglês'))
    file = form.file_uploader('Escolha o arquivo PDF:', type="pdf")
    submit_button = form.form_submit_button(label='✨ Resumir!')

    if file is not None and submit_button is not False:
        pdf = extract_data(file)
        with st.spinner('Resumindo...'):
            st.markdown(f'{display_sumarization(pdf, language)}',
                        unsafe_allow_html=True)
            #acc_sumarization(pdf, display_sumarization(pdf, language))
