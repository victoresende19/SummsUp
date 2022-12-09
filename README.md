# SuumsUp 📜

*SummsUp* é uma interface fácil de usar construída em Stramlit para criar resumos de textos digitados pelo usuário ou arquivos PDF. O aplicativo utiliza redes neurais pré-treinadas que aproveitam várias incorporações de NLP e depende de [Transformers](https://huggingface.co/transformers/).

# Línguas 🇧🇷 🇺🇲
- Português
- Inglês

# Redes neurais 🌐
Como explicado, o Summs Up utiliza redes neurais pré-treinadas das quais foram disponibilizadas no repositório [Hugging Face](https://huggingface.co/). Abaixo são detalhado as redes neurais utilizadas para a criação dos resumos.

- Português: [Portuguese T5 for Abstractive Summarization (PTT5 Summ)](https://huggingface.co/phpaiola/ptt5-base-summ-xlsum). O modelo foi ajustado por meio dos conjuntos de dados: WikiLingua, XL-Sum, TeMário e CSTNews.
- Inglês: [Bert-small2Bert-small Summarization with EncoderDecoder Framework (Bert-small2Bert-small)](https://huggingface.co/mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization). O modelo foi ajustado por meio dos conjuntos de dados: CNN e Dailymail.
