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

COLOR = {
    'LOC': '#40E0D0',
    'PER': '#6495ED',
    'ORG': '#CCCCFF',
    'MISC': '#FF7F50'
}

# Loading models


@st.cache(allow_output_mutation=True, show_spinner=True)
def load_models():
    tagger = SequenceTagger.load('flair/ner-english-fast')
    model = GBRT(EMB_PATH, model_path='coherence.pkl')
    model.two_step = True
    return model, tagger


# Page setup
st.set_page_config(layout="wide", page_title='Named Entity Disambiguation')
st.write("## Named Entity Disambiguation")
col1, col2 = st.columns(2)


def get_candidates(mentions_tags):
    candidates = []
    cache = {}
    for mention, tag in mentions_tags:
        if (mention, tag) in cache.keys():
            candidates.append((mention, cache[(mention, tag)]))
        else:
            try:
                res1 = google_search(f'{mention} {TYPE[tag]}')
            except:
                res1 = []
            res2 = wikipedia_search(mention)
            cands = list(set(res1 + res2))
            candidates.append((mention, cands))
            cache[(mention, tag)] = cands
    return candidates


def display_tag(text, typ, label, info):
    if label != 'NIL':
        label = "https://en.wikipedia.org/wiki/" + label
    return f"""
    <a style="margin: 0 5px; padding: 2px 4px; border-radius: 4px; text-decoration:none;
              background-color:{COLOR[typ]}; color: white; cursor:pointer" 
       title="{info}" href={label} target="_blank">
        <span style="padding-right:3px">{text}</span>
        <span>{typ}</span>
    </a>"""


def main(text):
    doc = Sentence(text)
    tagger.predict(doc)
    tagged, last_pos = '', 0

    with st.spinner('Generating candidates...'):
        mentions_cands = get_candidates(
            [(ent.text, ent.tag) for ent in doc.get_spans('ner')])

    with st.spinner('Disambiguating mentions...'):
        preditions = model.link(mentions_cands, text)

    with st.spinner('Rendering results...'):
        ent_desc = {}
        for i, ent in enumerate(doc.get_spans('ner')):
            label = preditions[i][1]
            if label not in ent_desc.keys():
                ent_desc[label] = get_entity_extract(label)
            tag = display_tag(ent.text, ent.tag, label, ent_desc[label])
            tagged += text[last_pos:ent.start_pos] + tag
            last_pos = ent.end_pos
        tagged += text[last_pos:]

    with col2:
        st.write("### Disambiguated Text")
        components.html(f'<p style="line-height: 1.8; margin-top:30px;">{tagged}</p>',
                        scrolling=True, height=400)

    df = pd.DataFrame(data=preditions, columns=['Mention', 'Prediction', 'Confidence'])
    st.write("**Additional Information**")
    st.dataframe(df)


if __name__ == '__main__':
    default_text = 'George Washington went to Washington.'
    model, tagger = load_models()

    with col1:
        st.write("### Input Text")
        user_input = st.text_area('Press Ctrl + Enter to update results', default_text,
                                  height=350)
        if user_input:
            main(user_input)
