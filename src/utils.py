import os
import json
import requests

import numpy as np
import pandas as pd
from src.env import CSE_API_KEY, CSE_CX


CWD = os.path.dirname(os.path.dirname(__file__))
AIDADIR = os.path.join(CWD, 'data', 'aida')


response = requests.get('https://google.com')
if response.status_code == 200:
    g_cookies = response.cookies.get_dict()


def load_json(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        return json.load(f)


def cos_sim(v1, v2):
    v1v2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    if v1v2 == 0:
        return 0
    else:
        return np.dot(v2, v1) / v1v2


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

    res = requests.get(service_url, params=params).json()['query']['pages']
    res = res[list(res.keys())[0]]
    extract = res['extract'] if 'extract' in res.keys() else ''
    return extract


"""
Custom Search Engine
Dashboard: https://cse.google.com/
JSON API Docs : https://developers.google.com/custom-search/v1/site_restricted_api 
"""
def google_search(query, num_results=5):
    service_url = "https://www.googleapis.com/customsearch/v1/siterestrict"
    params = {
        'q': query,
        'num': num_results,
        'start': 0,
        'key': CSE_API_KEY,
        'cx': CSE_CX 
    }
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
    res = requests.get(service_url, params=params, headers=headers, cookies=g_cookies)
    try:
        cands = [i['title'].replace(' - Wikipedia', '') for i in res.json()["items"]]
    except:
        print(f"An error occurred, returning a empty list of candidates.\nStatus code: {res.status_code}")
        cands = []
    return [i.replace(' ', '_') for i in cands]


def wikipedia_search(query, num_results=20):
    service_url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'opensearch',
        'search': query,
        'namespace': 0,
        'limit': num_results,
        'redirects': 'resolve',
    }

    results = requests.get(service_url, params=params).json()[1]
    results = [i.replace(' ', '_') for i in results if 'disambiguation' not in i.lower()]
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
