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
from PIL import Image


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


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False, ttl=24*3600, max_entries=2)
def portuguese_model():
    token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
    model_name = 'phpaiola/ptt5-base-summ-xlsum'

    tokenizer = T5Tokenizer.from_pretrained(token_name)
    model_pt = T5ForConditionalGeneration.from_pretrained(model_name)

    return tokenizer, model_pt


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False, ttl=24*3600, max_entries=2)
def english_model():
    #summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarizer = pipeline(
        "summarization", model="philschmid/bart-large-cnn-samsum")

    return summarizer


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False, ttl=24*3600, max_entries=2)
def ROUGE_metric():
    ROUGE = evaluate.load('rouge')

    return ROUGE


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False, ttl=24*3600, max_entries=2)
def portuguese_summarization(text: str) -> str:
    """
    Sumariza o texto disponibilizado (em português)
    recebe - texto: texto disponibilizado 
    retorna - texto: texto sumarizado (em português)
    """

    tokenizer, model_pt = portuguese_model()
    inputs = tokenizer.encode(text, max_length=512,
                              truncation=True, return_tensors='pt')
    summary_ids = model_pt.generate(
        inputs, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0])

    return summary


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False, ttl=24*3600, max_entries=2)
def english_summarization(text: str) -> str:
    """
    Sumariza o texto disponibilizado (em inglês)
    recebe - texto: texto disponibilizado 
    retorna - texto: texto sumarizado (em inglês)
    """

    summarizer = english_model()
    return summarizer(text)


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False, ttl=24*3600, max_entries=2)
def acc_summarization(texto: str, resumo: str) -> str:
    """
    Retorna a acurácia do resumo por meio da métrica ROUGE.
    recebe - texto: texto disponibilizado, resumo: texto resumido pelo modelo
    retorna - acuracia: acuracia do resumo
    """

    ROUGE = ROUGE_metric()

    texto_cru = [texto]
    texto_resumido = [resumo]
    acuracia = ROUGE.compute(references=texto_cru, predictions=texto_resumido)

    return acuracia


def display_summarization(text, language):
    if language == 'Português':
        return portuguese_summarization(text)
    elif language == 'Inglês':
        return english_summarization(text)


st.set_page_config(
    page_icon='🎈', page_title='Sumarizador de textos', layout='wide')
st.markdown("<h1 style='text-align: center; color: black; font-size: 42px'> 📋 Sumarizador de textos 📋 </h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black;'> Por Victor Augusto Souza Resende </p>",
            unsafe_allow_html=True)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

expanderAbout = st.sidebar.expander(
    label="🛈 Sobre o aplicativo", expanded=True)
expanderAbout.markdown(
    """
        - O *Sumarizador de Textos* é uma interface fácil de usar construída em Stramlit para criar resumos de textos digitados pelo usuário ou arquivos PDF.
        - O aplicativo utiliza redes neurais pré-treinadas que aproveitam várias incorporações de NLP e depende de [Transformers](https://huggingface.co/transformers/).
        - Além disso, a aplicação conta com suporte para resumir dois tipos de idiomas: Português e Inglês! 🤗
    """
)

st.sidebar.markdown('')
st.sidebar.markdown('')

st.sidebar.markdown(
    "<h4 style='text-align: center; color: black;'> Contate o autor por meio do QRCode </h4>", unsafe_allow_html=True)
st.sidebar.image(Image.open('Images/QRCode.png'),
                 caption='LinkedIn Victor Resende', width=230)


text_type = st.selectbox('Que maneira gostaria de resumir seu texto?',
                         ('Escolha as opções', 'Resumo escrito', 'Resumo em PDF'))

if text_type == 'Resumo escrito':
    form = st.form(key='my_form')
    language = form.selectbox('Qual a língua do PDF?', ('Português', 'Inglês'))
    text = form.text_area(
        "Texto a ser resumido:",
        height=300,
        placeholder='Escreva aqui...'
    )
    submit_button = form.form_submit_button(label='✨ Resumir!')

    if submit_button:
        with st.spinner('Resumindo...'):
            final_summary = display_summarization(text, language)
            st.markdown(
                "<h4 style='text-align: center; color: black;'> Resumo </h4>",  unsafe_allow_html=True)
            st.info(
                f"{final_summary.replace('<pad> ', '').replace('</s>', '')}")
            st.markdown(
                f"<p> Acurácia (<a href='https://huggingface.co/spaces/evaluate-metric/rouge'>ROUGE</a>): {acc_summarization(text, final_summary)}</p>", unsafe_allow_html=True)

elif text_type == 'Resumo em PDF':
    form = st.form(key='my_form')
    language = form.selectbox('Qual a língua do PDF?', ('Português', 'Inglês'))
    file = form.file_uploader('Escolha o arquivo PDF:', type="pdf")
    submit_button = form.form_submit_button(label='✨ Resumir!')

    if file is not None and submit_button is not False:
        pdf = extract_data(file)
        with st.spinner('Resumindo...'):
            final_summary = display_summarization(pdf, language)
            st.markdown(
                "<h4 style='text-align: center; color: black;'> Resumo </h4>", unsafe_allow_html=True)
            st.info(
                f"{final_summary.replace('<pad> ', '').replace('</s>', '')}")
            st.markdown(
                f"<p> Acurácia (<a href='https://huggingface.co/spaces/evaluate-metric/rouge'>ROUGE</a>): {acc_summarization(pdf, final_summary)}</p>", unsafe_allow_html=True)
