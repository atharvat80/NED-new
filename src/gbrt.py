import os
import pickle

import numpy as np
from nltk import edit_distance

from src.base import Base
from src.utils import cos_sim


CWD = os.path.dirname(os.path.dirname(__file__))
DATADIR = os.path.join(CWD, 'data', 'GBRT')

"""
Load Datafiles
"""
with open(os.path.join(DATADIR, 'entity_anchors.pkl'), 'rb') as f:
    prior_prob = pickle.load(f)
with open(os.path.join(DATADIR, 'entity_prior.pkl'), 'rb') as f:
    entity_prior = pickle.load(f)


"""
Helper Functions
"""
def get_edit_dist(x, y):
    return edit_distance(x, y)


def get_entity_prior(entity):
    try:
        return entity_prior[entity.replace('_', ' ')]
    except:
        return 0


def get_prior_prob(entity, mention):
    try:
        entity = entity.replace('_', ' ')
        mention = mention.lower()
        return prior_prob[mention][entity] / sum(prior_prob[mention].values())
    except:
        return 0


def get_max_prior_prob(mentions, candidates):
    max_prob = {i: max([get_prior_prob(i, j) for j in mentions])
                for i in candidates}
    return max_prob


def load_model(fname):
    model = None
    with open(os.path.join(DATADIR, fname), 'rb') as f:
        model = pickle.load(f)
    return model


"""
Entity Ranking using Gradient Boosting Regression Trees
"""
class GBRT(Base):
    def __init__(self, emb_path, model_path=None, two_step=False, cased=False, nouns_only=True):
        super().__init__(emb_path, cased=cased, nouns_only=nouns_only)
        self.two_step = two_step
        if model_path is not None:
            self.model = load_model(model_path)

    def encode_context_entities(self, context_entities):
        emb, n = np.zeros(self.vector_size), 1
        for i in context_entities:
            emb += self.encode_entity(i)
            n += 1
        return emb/n

    def link(self, mentions_cands, context):
        n_features = self.model.n_features_in_

        # Calculate max prior probability of all candidates.
        mentions = set([i for i, _ in mentions_cands])
        candidates = set([i for _, j in mentions_cands for i in j])
        max_prob = get_max_prior_prob(mentions, candidates)

        # Find unambiguous entities
        unamb_entities = [x for i, j in mentions_cands
                          for x in j if get_prior_prob(x, i) > 0.95]
        context_ent_emb = self.encode_context_entities(unamb_entities)

        # Make predictions
        context_emb = self.encode_sentence(context)
        predictions = []
        for mention, candidates in mentions_cands:
            # Generate feature values
            num_cands = len(candidates)
            X = []
            for candidate in candidates:
                cand = candidate.replace('_', ' ').lower()
                ment = mention.lower()
                cand_emb = self.encode_entity(candidate)

                # At the minimum add base feature values
                feat_values = [
                    candidate,
                    get_prior_prob(candidate, mention),
                    get_entity_prior(candidate),
                    max_prob[candidate],
                    num_cands
                ]
                # Add string similarity
                if n_features >= 8:
                    feat_values += [
                        get_edit_dist(ment, cand),
                        ment == cand,
                        ment in cand,
                        cand.startswith(cand) or cand.endswith(ment)
                    ]
                # Add context similarity
                if n_features >= 9:
                    feat_values.append(cos_sim(cand_emb, context_emb))
                # Add coherence score
                if n_features >= 10:
                    feat_values.append(cos_sim(cand_emb, context_ent_emb))
                X.append(feat_values)

            # Add rank
            if n_features == 11:
                X.sort(key=lambda x: x[-1] + x[-2], reverse=True)
                X = [j + [i + 1] for i, j in enumerate(X)]

            # Predict
            pred, conf = 'NIL', 0
            for i in X:
                c = self.model.predict(np.array([i[1:]]))[0]
                if c > conf:
                    pred = i[0]
                    conf = c
            predictions.append([mention, pred, conf])

            # Update context entity embedding (two-step)
            if self.two_step and n_features >= 10:
                context_ent_emb += self.encode_entity(pred)
                context_ent_emb /= 2

        return predictions
