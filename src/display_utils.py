import os
import streamlit as st
import base64
import pymorphy3
import re

# background
def set_page_bg(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: contain;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    return


def highlight_words(text, words):
        morph = pymorphy3.MorphAnalyzer()
        tokens = text.split("\n")  # Разделяем текст по переносам строк
        highlighted_lines = []
        for line in tokens:
            line_tokens = re.findall(r'\b\w+\b|[^\w\s]', line, re.UNICODE)
            highlighted_line = ""
            for token in line_tokens:
                parsed_token = morph.parse(token)[0]
                normal_form = parsed_token.normal_form
                if normal_form in words:
                    highlighted_line += f"<span style='background-color: #fecedc'>{token}</span> "
                else:
                    highlighted_line += f"{token} "
            highlighted_lines.append(highlighted_line.strip())
        return "<br>".join(highlighted_lines)


def load_words(filename):
    with open(os.getcwd() + filename, 'r') as f:
        words = f.read().splitlines()
        f.close()
        return words