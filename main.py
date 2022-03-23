import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from src.gbrt import GBRT, prior_prob
from src.utils import wikipedia_search, google_search, get_entity_extract

EMB_PATH = os.path.join(os.getcwd(), 'embeddings', 'wiki2vec_w10_100d.pkl')

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
    # NER
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    bert_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    tagger = pipeline("token-classification", model=bert_ner, tokenizer=tokenizer, 
                      device=0, aggregation_strategy="simple")
    # NED
    model = GBRT(EMB_PATH, model_path='wiki2vec_w10_100d.pkl_trained.pkl', two_step=True)
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
            res1 = google_search(mention)
            res2 = wikipedia_search(mention, limit=30)
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
    ner_results = tagger(text)
    tagged, last_pos = '', 0

    with st.spinner('Generating candidates...'):
        mentions_cands = get_candidates([(res['word'], res['entity_group']) for res in ner_results])

    with st.spinner('Disambiguating mentions...'):
        preditions = model.link(mentions_cands, text)

    with st.spinner('Rendering results...'):
        ent_desc = {}
        for i, res in enumerate(ner_results):
            label = preditions[i][1]
            if label not in ent_desc.keys():
                ent_desc[label] = get_entity_extract(label)
            tag = display_tag(res['word'], res['entity_group'], label, ent_desc[label])
            tagged += text[last_pos:res['start']] + tag
            last_pos = res['end']
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
        user_input = st.text_area('Press Ctrl + Enter to update results', default_text, height=350)
        if user_input:
            main(user_input)
