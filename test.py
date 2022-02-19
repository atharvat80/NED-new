import os
import pandas as pd

from tqdm import tqdm
from src.utils import AIDADIR, get_document, get_local_context, load_json


CWD = os.path.dirname(__file__)
AIDADIR = os.path.join(CWD, 'data', 'aida')


def aida_local(model):
    correct = 0
    predictions = []
    model.entity_desc_dict = load_json(os.path.join(AIDADIR, 'entities.json'))
    # For each document
    for doc in tqdm(range(1163, 1394)):
        df = pd.read_csv(os.path.join(AIDADIR, 'candidates', f'{doc}.csv'))
        mentions_tags = df[['mention', 'tag']].drop_duplicates().to_numpy()
        # For each mention in a document
        for mention, tag in mentions_tags:
            candidates = df[(df['mention'] == mention) & (df['tag'] == tag)]
            candidates = candidates['candidate'].to_numpy()
            context = get_local_context(mention, tag, doc)
            # If the mention has a valid tag
            if context and tag != 'NIL':
                pred, conf = model.link(mention, context, candidates)
                predictions.append([mention, tag, pred])
                if pred == tag:
                    correct += 1
    
    accuracy = round((correct/len(predictions))*100, 3)
    results  = pd.DataFrame(predictions, columns=['mention', 'tag', 'pred'])
    return accuracy, results
                

def aida_global(model):
    total = 0
    correct = 0
    predictions = []
    model.entity_desc_dict = load_json(os.path.join(AIDADIR, 'entities.json'))
    
    for doc in tqdm(range(1163, 1394)):
        # Generate test data
        df = pd.read_csv(os.path.join(AIDADIR, 'candidates', f'{doc}.csv'))
        mentions_tags = df[['mention', 'tag']].drop_duplicates().to_numpy()
        mentions_cands = []
        for mention, tag in mentions_tags:
            cands = df[(df['mention'] == mention) & (df['tag'] == tag)]
            cands = cands['candidate'].to_numpy()
            if tag != 'NIL':
                total += 1
                mentions_cands.append([mention, cands])
        
        # make predictions and compare to ground truth
        j = 0
        context = get_document(doc)
        preds   = model.link(mentions_cands, context)
        for mention, tag in mentions_tags:
            if j >= len(preds):
                break
            m, pred, _ = preds[j]
            if m == mention:
                predictions.append([m, tag, pred])
                j += 1
                if pred == tag:
                    correct += 1

    accuracy = round((correct/total)*100, 3)
    results = pd.DataFrame(predictions, columns=['mention', 'tag', 'pred'])
    return accuracy, results