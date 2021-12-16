import os

import flair.datasets as datasets
import pandas as pd
from tqdm import tqdm


# Define data path
CWD = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATADIR = os.path.join(CWD, 'data', 'aida')

# Load necessary data
print("Loading AIDA dataset...")
aida = datasets.NEL_ENGLISH_AIDA()
aida_sets = {
    'train': aida.train,
    'test': aida.test,
    'dev': aida.dev
}

# Get description of test entities
test_entity_desc = pd.read_csv(os.path.join(DATADIR, 'entity_desc', 'test.csv'))


def get_document(num):
    with open(os.path.join(DATADIR, 'docs', f'{num}.txt'), 'r') as f:
        return f.read().replace('\n', ' ')


def get_candidates(doc, mention, tag):
    df = pd.read_csv(os.path.join(DATADIR, 'candidates', f'{doc}.csv'))
    df = df[(df['mention'] == mention) & (df['tag'] == tag)]
    return df['candidate'].to_numpy()


def get_test_entity_desc(entity):
    try:
        return test_entity_desc[test_entity_desc['entity'] == entity]['description'].values[0]
    except:
        return ''


def get_test_data(set_):
    mentions_tags = []
    doc = 0 if set_ == 'train' else 946 if set_ == 'dev' else 1162
    for i in aida_sets[set_]:
        context = i.to_plain_string()
        if context != '-DOCSTART-':
            mentions_tags += [[j.text, j.tag, context, doc]
                              for j in i.get_spans()]
        else:
            doc += 1
    return mentions_tags


def test(model, global_context=False, inc_desc=True):
    mentions_tags = get_test_data('test')
    preds = []
    for mention, tag, local_context, doc_num in tqdm(mentions_tags):
        candidates = get_candidates(doc_num, mention, tag)
        if tag in candidates:
            if inc_desc:
                candidates = [[i, get_test_entity_desc(i)] for i in candidates]
            if global_context:
                document_text = get_document(doc_num)
                pred, _ = model.link(mention, document_text, candidates)
            else:
                pred, _ = model.link(mention, local_context, candidates)
            preds.append([mention, tag, pred])

    accuracy = round((sum([1 for _, t, p in preds if t == p]) / len(preds)) * 100, 2)
    print(f'Accuracy: {accuracy}%\nTotal test samples: {len(preds)}')
