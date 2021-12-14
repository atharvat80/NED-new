import os
import pickle

import nltk

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