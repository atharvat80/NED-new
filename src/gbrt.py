import os
import pickle

import nltk
import numpy as np

from src.base import Base, BaseWiki2Vec
from src.utils import cos_sim

# Define data path
CWD = os.path.dirname(os.path.dirname(__file__))
DATADIR = os.path.join(CWD, 'data', 'GBRT')

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


class GBRT(BaseWiki2Vec):
    def __init__(self, emb_path, model_path=None, two_step=False):
        super().__init__(emb_path)
        if model_path is not None:
            self.model = load_model(model_path)
        self.two_step = two_step

    def get_nouns(self, s):
        nouns = []
        for word, pos in nltk.pos_tag(self.tokenizer(s)):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.extend(word.split(' '))
        return list(set(nouns))

    def encode_sentence(self, s):
        nouns = [i for i in self.get_nouns(s.lower())]
        emb, n = np.zeros(self.vector_size), 1
        for i in nouns:
            if self.emb.get_word(i) is not None:
                emb += self.emb.get_word_vector(i)
                n += 1
        return emb/n

    def encode_context_entities(self, context_entities):
        initial = np.zeros(self.vector_size)
        for i in context_entities:
            initial += self.encode_entity(i)
        return initial/(len(context_entities) + 1)

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


class GBRT2(Base):
    def __init__(self, emb_path, model_path=None, cased=False):
        super().__init__(emb_path, cased=cased)
        if model_path is not None:
            self.model = load_model(model_path)
        self.two_step = False
        self.vector_size = self.emb.vector_size

    # def get_nouns(self, s):
    #     nouns = []
    #     for word, pos in nltk.pos_tag(self.tokenizer(s)):
    #         if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
    #             nouns.extend(word.split(' '))
    #     return list(set(nouns))

    # def encode_sentence(self, s):
    #     nouns = [i for i in self.get_nouns(s.lower())]
    #     emb, n = np.zeros(self.vector_size), 1
    #     for i in nouns:
    #         try:
    #             emb += self.emb.get_vector(i)
    #         except:
    #             pass   
    #     return emb/n

    def encode_context_entities(self, context_entities):
        initial = np.zeros(self.vector_size)
        for i in context_entities:
            initial += self.encode_entity(i)
        return initial/(len(context_entities) + 1)

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
