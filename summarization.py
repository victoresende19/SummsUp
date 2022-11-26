# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:18:08 2022

@author: Victor Resende
"""
import streamlit as st
import pdfplumber
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import evaluate

HARIM = evaluate.load('NCSOFT/harim_plus')

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
    
    inputs = tokenizer.encode(text, max_length=512, truncation=True, return_tensors='pt')
    summary_ids = model_pt.generate(inputs, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0])

    return write_html(summary)

def english_sumarization(text):
    """'
    Sumariza o texto disponibilizado (em inglês)
    
    recebe - texto: texto disponibilizado 
    retorna - texto: texto sumarizado (em inglês)
    """
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

def acc_sumarization(texto: str, resumo: str) -> str:
    """'
    Retorna a acurácia do resumo por meio da métrica Harim.
    
    recebe - texto: texto disponibilizado, resumo: texto resumido pelo modelo
    retorna - acuracia: acuracia do resumo
    """
    
    texto_cru = [texto]
    texto_resumido = [resumo]
    acuracia = HARIM.compute(references = texto_cru, predictions = texto_resumido)
    
    return round(acuracia[0], 4)


def display_sumarization(text, language):
    if language == 'Português':
        return write_html(f"<h2 style='text-align: center; color: black;'> Texto sumarizado: </h2> <br><br> {portuguese_sumarization(text)}")
    elif language == 'Inglês':
        return write_html(f"<h2 style='text-align: center; color: black;'> Texto sumarizado: </h2> <br><br> {english_sumarization(text)}")


st.set_page_config(page_icon = '📋', page_title = 'Sumarizador de textos')
write_html("<h1 style='text-align: center; color: black;'> Sumarizador de textos </h1>")
write_html("<p align='justify'> Por Victor Augusto Souza Resende  <p align='justify'>")

text_type = st.selectbox('Que maneira gostaria de resumir seu texto?',('Escolha as opções', 'Escrevendo', 'PDF'))

if text_type == 'Escrevendo':
    form = st.form(key='my_form')
    language = form.selectbox('Qual a língua do PDF?',('Português', 'Inglês'))
    text = form.text_input('Texto a ser resumido:', placeholder='Escreva aqui...')
    submit_button = form.form_submit_button(label='Aplicar')
    
    if submit_button:
        with st.spinner('Resumindo...'):
            display_sumarization(text, language)
    
elif text_type == 'PDF':
    form = st.form(key='my_form')
    language = form.selectbox('Qual a língua do PDF?',('Português', 'Inglês'))
    file = form.file_uploader('Escolha o arquivo PDF:', type="pdf")
    submit_button = form.form_submit_button(label='Aplicar')

    if file is not None and submit_button is not False:
        pdf = extract_data(file)
        with st.spinner('Resumindo...'):
            display_sumarization(pdf, language)