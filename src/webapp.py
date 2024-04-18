import sys
import os
import streamlit as st
model_name = 'roberta-base'
from display_utils import set_page_bg, highlight_words, load_words
sys.path.insert(0, '../src')
import json
import pandas as pd
import plotly.express as px
import requests
from clean_text import text_preprocessing_pipeline
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# load labels
with open(os.getcwd() + '/model/labels.txt', 'r') as f:
    labels = f.read().splitlines()
    f.close()


# load model
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l1 = torch.nn.Linear(768, 256)
        self.fc = torch.nn.Linear(768,5)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        #features = F.relu(self.l1(features))
        # output_2 = self.l2(output_1)
        output = F.softmax(self.fc(features), dim=1)
        return output
model = BERTClass()
model.load_state_dict(torch.load(os.getcwd() + '/model/model4.bin', map_location=torch.device('cpu')))
vectorizer = AutoTokenizer.from_pretrained(model_name)


def predict(text):
    text = text_preprocessing_pipeline(text)
    text = vectorizer.encode_plus(text,
                                  truncation=True,
                                  add_special_tokens=True,
                                  max_length=200,
                                  padding='max_length',
                                  return_token_type_ids=True
                                  )
    input_ids = torch.tensor(text['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(text['attention_mask']).unsqueeze(0)
    token_type_ids = torch.tensor(text['token_type_ids']).unsqueeze(0)
    output = model(input_ids, attention_mask, token_type_ids).detach().numpy()
    return labels[output.argmax().item()]


DATA = './data/top_words.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA)


# draw a map
def draw_map_cases():
    fig = px.choropleth_mapbox(df,
                               geojson=json_locations,
                               locations='iso_code',
                               hover_data=['top_word'],
                               color_continuous_scale="Reds",
                               mapbox_style="carto-positron",
                               title="Most frequent words in country",
                               zoom=1,
                               opacity=0.5,
                               )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

# set background
set_page_bg(os.getcwd() + '/img/bg.png')

with open('data/countries.geo.json') as json_file:
    json_locations = json.load(json_file)
# Draw the map
df = load_data()

st.sidebar.title("Выберите функцию для отображения")

select_event = st.sidebar.selectbox('', ('Жанровый классификатор', 'Интерактивная карта'))
if select_event == 'Жанровый классификатор':
    st.markdown("<h1 style='text-align: center; color: #322c2c;'>Жанровый классификатор</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color: #fe6053;'>Введите текст песни</div>", unsafe_allow_html=True)
    text = open('src/lyrics/1.txt', 'r').read()
    lyrics = st.text_area("", value=text, height=500)

    # display the name when the submit button is clicked
    # .title() is used to get the input text string
    if(st.button('Submit')):
        genre = predict(lyrics)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h2 style='color:black'>Жанр: <span style='color:red'>{genre}</span></h2>", unsafe_allow_html=True)

        st.balloons()

        if(genre == 'rhythm and blues'):
            genre = 'rb'

        css_style = """
            <style>
                body {
                    font-size: 16px;
                    line-height: 1.3;
                }
            </style>
        """

        st.markdown(css_style, unsafe_allow_html=True)


        # Список слов для выделения
        words_to_highlight_rap = load_words('/src/popular_rap_words.txt')
        words_to_highlight_metal = load_words('/src/popular_metal_words.txt')
        words_to_highlight_rock = load_words('/src/popular_rock_words.txt')
        words_to_highlight_pop = load_words('/src/popular_pop_words.txt')
        words_to_highlight_rb = load_words('/src/popular_rb_words.txt')
        genre_to_list = {"rap": words_to_highlight_rap, "metal": words_to_highlight_metal,
                         "rock": words_to_highlight_rock, "pop": words_to_highlight_pop,
                         "rb": words_to_highlight_rb}

        # Выделение слов в тексте
        words_to_highlight = genre_to_list[genre]
        highlighted_text = highlight_words(lyrics, words_to_highlight)

        # Отображение текста с выделенными словами
        st.markdown("<h2 style='color:black'>Popular words for this genre:</h2>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:black'>{highlighted_text}</span>", unsafe_allow_html=True)


if select_event == 'Интерактивная карта':
    st.markdown("<h1 style='text-align: center; color: #322c2c;'>Самые популярные слова в треках разных стран</h1>", unsafe_allow_html=True)
    # st.title('Самые популярные слова в странах мира')
    st.plotly_chart(draw_map_cases(), use_container_width=True)
