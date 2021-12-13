import json
import os
import pickle
from urllib import parse, request

import nltk
from googlesearch import search

# Define data path
CWD = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATADIR = os.path.join(CWD, 'data')

# Load required data files
prior_prob = None
entity_prior = None
with open(os.path.join(DATADIR, 'entity_anchors.pkl'), 'rb') as f:
    prior_prob = pickle.load(f)
with open(os.path.join(DATADIR, 'entity_prior.pkl'), 'rb') as f:
    entity_prior = pickle.load(f)


def make_request(service_url, params):
    url = service_url + '?' + parse.urlencode(params)
    response = json.loads(request.urlopen(url).read())
    return response


def get_edit_dist(x, y):
    return nltk.edit_distance(x, y)


def get_entity_prior(entity):
    try:
        return entity_prior[entity]
    except:
        return 0


def get_prior_prob(entity, mention):
    try:
        mention = mention.lower()
        return prior_prob[mention][entity] / sum(prior_prob[mention].values())
    except:
        return 0


def get_entity_info(entity_title, num_sentences=1):
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


def wiki_data_search(query, num_results=10):
    # Get WikiData entities from wikidata
    service_url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'search': query,
        'language': 'en',
        'format': 'json',
        'limit': num_results,
    }
    response = make_request(service_url, params)
    results = {}
    for i in response['search']:
        if 'description' in i.keys() and 'disambiguation' not in i['description']:
            results[i['id']] = {'label': i['label'], 'description': i['description']}

    # add wikipedia url to results
    params = {
        'action': 'wbgetentities',
        'ids': '|'.join([i for i in results.keys()]),
        'props': 'sitelinks',
        'sitefilter': 'enwiki',
        'format': 'json',
    }
    response = make_request(service_url, params)
    if 'entities' in response.keys():
        to_remove = []
        for i in results.keys():
            try:
                results[i]['label'] = response['entities'][i]['sitelinks']['enwiki']['title']
            except:
                to_remove.append(i)
        for i in to_remove:
            del results[i]
        return [results[i] for i in results.keys()]
    else:
        return []


def get_page_links(entity):
    def parse_response(data):
        pages = data["query"]["pages"]
        page_titles = []
        for _, val in pages.items():
            for link in val["links"]:
                page_titles.append(link["title"])
        return page_titles

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": entity,
        "prop": "links",
        "pllimit": "max",
        "plnamespace": 0
    }
    # make the initial request
    data = make_request(url, params)
    page_titles = parse_response(data)
    # keep going until the last page
    while "continue" in data:
        plcontinue = data["continue"]["plcontinue"]
        params["plcontinue"] = plcontinue
        data = make_request(url, params)
        page_titles += parse_response(data)

    return page_titles

# ---------------------------------------
#  references
# ---------------------------------------
# https://stackoverflow.com/questions/27452656/wikidata-entity-value-from-name
# https://www.wikidata.org/w/api.php?action=help&modules=wbsearchentities
# https://en.wikipedia.org/w/api.php?action=query&redirects=1&titles=Japan+national+football+team&format=json&prop=extracts|description|pageprops&explaintext=1&exsectionformat=plain&exsentences=2
# https://en.wikipedia.org/w/api.php?action=opensearch&search=Japan+Football&namespace=0&limit=20&redirects=resolve
