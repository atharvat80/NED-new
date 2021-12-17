import json
import os
from urllib import parse, request

import numpy as np
import pandas as pd
from googlesearch import search
from tqdm import tqdm

CWD = os.path.dirname(os.path.dirname(__file__))
AIDADIR = os.path.join(CWD, 'data', 'aida')


def cos_sim(v1, v2):
    v1v2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    if v1v2 == 0:
        return 0
    else:
        return np.dot(v2, v1) / v1v2


def make_request(service_url, params):
    url = service_url + '?' + parse.urlencode(params)
    response = json.loads(request.urlopen(url).read())
    return response


def get_entity_extract(entity_title, num_sentences=1):
    service_url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'titles': entity_title,
        'prop': 'extracts',
        'redirects': 1,
        'format': 'json',
        'explaintext': 1,
        'exsectionformat': 'plain',
        'exsentences': num_sentences
    }

    res = make_request(service_url, params)['query']['pages']
    res = res[list(res.keys())[0]]
    extract = res['extract'] if 'extract' in res.keys() else ''
    return extract


def google_search(query, num_results=10):
    results = search(f"{query} site:en.wikipedia.org", num_results=num_results)
    return [i[30:] for i in results]


def wikipedia_search(query, num_results=20):
    service_url = 'https://en.wikipedia.org/w/api.php'
    search_params = {
        'action': 'opensearch',
        'search': query,
        'namespace': 0,
        'limit': num_results,
        'redirects': 'resolve',
    }

    results = make_request(service_url, search_params)[1]
    results = [i.replace(' ', '_')
               for i in results if 'disambiguation' not in i.lower()]
    return results


def get_document(i):
    text = ''
    with open(os.path.join(AIDADIR, 'docs', f'{i}.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_local_context(mention, tag, doc):
    df = pd.read_csv(os.path.join(AIDADIR, 'local_context.csv'))
    res = df[(df['mention'] == mention) & (df['tag'] == tag) & (df['doc'] == doc)]
    try:
        return res['context'].values[0]
    except:
        return ''


def test_local(model):
    correct = 0
    predictions = []
    model.entity_desc_df = pd.read_csv(os.path.join(AIDADIR, 'entity_desc.csv'))
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
                

def test_global(model):
    total = 0
    correct = 0
    predictions = []
    
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
