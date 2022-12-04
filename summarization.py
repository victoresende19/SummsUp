# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:18:08 2022

@author: Victor Resende
"""
import streamlit as st
import pdfplumber
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from io import StringIO
import re


def file_upload(file):
    if file is not None:
        pdf = extract_data(file)
        return pdf


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def extract_data(doc):
    pdf = pdfplumber.open(doc)
    text = ''
    for page in pdf.pages:
        text = text + page.extract_text()
        text = text + ' '
    return text


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def portuguese_summarization(text: str) -> str:
    """
    Sumariza o texto disponibilizado (em português)

    recebe - texto: texto disponibilizado 
    retorna - texto: texto sumarizado (em português)
    """

    def WHITESPACE_HANDLER(k): return re.sub(
        '\s+', ' ', re.sub('\n+', ' ', k.strip()))
    article_text = text

    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=256,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return summary


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def english_summarization(text: str) -> str:
    """
    Sumariza o texto disponibilizado (em inglês)

    recebe - texto: texto disponibilizado 
    retorna - texto: texto sumarizado (em inglês)
    """

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']


@st.cache(hash_funcs={StringIO: StringIO.getvalue}, allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def acc_summarization(texto: str, resumo: str) -> str:
    """
    Retorna a acurácia do resumo por meio da métrica Harim.

    recebe - texto: texto disponibilizado, resumo: texto resumido pelo modelo
    retorna - acuracia: acuracia do resumo
    """

    HARIM = evaluate.load('NCSOFT/harim_plus')

    texto_cru = [texto]
    texto_resumido = [resumo]
    acuracia = HARIM.compute(references=texto_cru, predictions=texto_resumido)

    return round(acuracia[0], 4)


def display_summarization(text, language):
    if language == 'Português':
        return portuguese_summarization(text)
    elif language == 'Inglês':
        return english_summarization(text)


st.set_page_config(page_icon='🎈', page_title='Sumarizador de textos', layout='wide')
st.markdown("<h1 style='text-align: center; color: black; font-size: 42px'> 📋 Sumarizador de textos 📋 </h1>",unsafe_allow_html=True)

st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
expander = st.sidebar.expander(label="🛈 Sobre o aplicativo", expanded=True)
#expander = st.expander(label="🛈 Sobre o aplicativo", expanded=True)
expander.markdown(
    """
        - O *Sumarizador de Textos* é uma interface fácil de usar construída em Stramlit para criar resumos de textos digitados pelo usuário ou arquivos PDF.
        - O aplicativo utiliza redes neurais pré-treinadas que aproveitam várias incorporações de NLP e depende de [Transformers](https://huggingface.co/transformers/).
        - Além disso, a aplicação conta com suporte para resumir dois tipos de idiomas: Português e Inglês! 🤗 
        - Para mais informações ou sugestões, contate o autor: [Victor Resende](https://www.linkedin.com/in/victor-resende-508b75196/). 
    """
)


text_type = st.selectbox('Que maneira gostaria de resumir seu texto?', ('Escolha as opções', 'Resumo escrito', 'Resumo em PDF'))

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
            st.markdown("<h4 style='text-align: center; color: black;'> Resumo </h4>",  unsafe_allow_html=True)
            st.info(f"{display_summarization(text, language).replace('<pad> ', '').replace('</s>', '')}")
            #st.markdown(f"<p> Acurácia (<a href='https://huggingface.co/spaces/NCSOFT/harim_plus'>HaRiM</a>): {acc_summarization(text, display_summarization(text, language))}</p>", unsafe_allow_html=True)

elif text_type == 'Resumo em PDF':
    form = st.form(key='my_form')
    language = form.selectbox('Qual a língua do PDF?', ('Português', 'Inglês'))
    file = form.file_uploader('Escolha o arquivo PDF:', type="pdf")
    submit_button = form.form_submit_button(label='✨ Resumir!')

    if file is not None and submit_button is not False:
        pdf = extract_data(file)
        with st.spinner('Resumindo...'):
            st.markdown("<h4 style='text-align: center; color: black;'> Resumo </h4>",  unsafe_allow_html=True)
            st.info(f"{display_summarization(pdf, language).replace('<pad> ', '').replace('</s>', '')}")
            #st.markdown(f"<p> Acurácia (<a href='https://huggingface.co/spaces/NCSOFT/harim_plus'>HaRiM</a>): {acc_summarization(pdf, display_summarization(pdf, language))}</p>", unsafe_allow_html=True)
