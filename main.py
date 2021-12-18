import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from flair.data import Sentence
from flair.models import SequenceTagger
from src.GBRT import GBRT
from src.utils import wikipedia_search, google_search, get_entity_extract

EMB_PATH = "C:\\Personal Files\\NED-using-KG\\embeddings\\wiki2vec_w10_100d.pkl"

TYPE = {
    'LOC': 'location',
    'PER': 'person',
    'ORG': 'organization',
    'MISC': ''
}

# Loading models
tagger = SequenceTagger.load('flair/ner-english-fast')
model = GBRT(EMB_PATH, model_path='coherence.pkl')
default_text = 'George Washington went to Washington.'

# Page setup
st.set_page_config(layout="wide", page_title='Named Entity Disambiguation')
st.write("## Named Entity Disambiguation")
col1, col2 = st.columns(2)


def get_candidates(query, tag):
    res1 = google_search(f'{query} {TYPE[tag]}')
    res2 = wikipedia_search(query)
    return list(set(res1 + res2))


def display_tag(text, label):
    info = 'unknown entity' if label == 'NIL' else get_entity_extract(label)
    if label != 'NIL':
        label = "https://en.wikipedia.org/wiki/" + label
    return f'<a style="padding: 2px 5px; background-color: #1ABC9C; \
        border-radius: 5px; color: white; cursor:pointer; text-decoration:none" \
        href={label} target="_blank" title="{info}">{text}</a>'


def main(text):
    doc = Sentence(text)
    tagger.predict(doc)
    tagged, last_pos = '', 0

    with st.spinner('Generating candidates...'):
        mentions_cands = [
            [ent.text, get_candidates(ent.text, ent.tag)]
            for ent in doc.get_spans('ner')
        ]

    with st.spinner('Disambiguating mentions...'):
        preditions = model.link(mentions_cands, text)

    with st.spinner('Rendering results...'):
        for i, ent in enumerate(doc.get_spans('ner')):
            tag = display_tag(ent.text, preditions[i][1])
            tagged += text[last_pos:ent.start_pos] + tag
            last_pos = ent.end_pos
        tagged += text[last_pos:]

    with col2:
        st.write("### Disambiguated Text")
        components.html(f'<p style="line-height: 1.8; margin-top:30px;">{tagged}</p>',
                        scrolling=True, height=350)

    df = pd.DataFrame(data=preditions, columns=['Mention', 'Prediction', 'Confidence'])
    st.write("**Additional Information**")
    st.dataframe(df)


if __name__ == '__main__':
    with col1:
        st.write("### Input Text")
        user_input = st.text_area('Press Ctrl + Enter to update results', default_text,
                                  height=350)
        if user_input:
            main(user_input)