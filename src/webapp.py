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


DATA = './data/top_words.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA)


# load labels
with open(os.getcwd() + '/model/labels.txt', 'r') as f:
    labels = f.read().splitlines()
    f.close()

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
        with open('./data/server.json') as json_file:
            server = json.load(json_file)
            responce = eval(requests.get('http://' + server['ip_address'] + '/predict/' + lyrics.title()).text)

        genre = responce['predict']
        title = responce['title']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h2 style='color:black'>Жанр: <span style='color:red'>{genre}</span></h2>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h2 style='color:black'>Сгенерированное название: <span style='color:red'>{title}</span></h2>", unsafe_allow_html=True)
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
    st.plotly_chart(draw_map_cases(),
                    use_container_width=True
                    )

