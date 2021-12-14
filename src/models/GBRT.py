import os
import pickle

import nltk
import numpy as np
from src.models.base import BaseWiki2Vec, cos_sim
from src.utils.GBRT import get_edit_dist, get_entity_prior, get_prior_prob

# Define data path
CWD = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATADIR = os.path.join(CWD, 'data')


class GBRT(BaseWiki2Vec):
    def __init__(self, emb, filter_stopwords=True, vector_size=100):
        super().__init__(emb, filter_stopwords=filter_stopwords, vector_size=vector_size)

    def get_nouns(self, s):
        s = self.tokenizer(s)
        nouns = []
        for word, pos in nltk.pos_tag(s):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.extend(word.split(' '))
        return list(set(nouns))

    def encode_sentence(self, s):
        s = s.lower()
        nouns = [i for i in self.get_nouns(s)]
        embs = []
        for i in nouns:
            if self.emb.get_word(i) is not None:
                embs.append(self.emb.get_word_vector(i))
        return sum(embs)/len(embs) if embs else np.zeros(self.vector_size)

    def get_unambiguous_entities(self, mentions_cands, threshold=0.95):
        unamb_entities = []
        for mention in mentions_cands:
            for candidate in mentions_cands[mention]:
                prior_prob = get_prior_prob(candidate, mention)
                if prior_prob >= threshold:
                    unamb_entities.append(candidate)
        return list(set(unamb_entities))

    def get_context_entity_emb(self, context_entites, initial=None):
        j = 0
        if initial is None:
            initial = np.zeros(np.zeros(self.vector_size))
        for i in context_entites:
            i = i.replace('_', ' ')
            if self.emb.get_entity(i):
                initial += self.emb.get_entity_vector(i)
                j += 1
        return initial/j

    def context_sim(self, context, candidates):
        scores = {}
        context_emb = self.encode_sentence(context)
        candidate_emb = {i: self.encode_entity(i) for i in candidates}
        for i in candidate_emb:
            scores[i] = cos_sim(context_emb, candidate_emb[i])
        return scores

    def rank(self, mentions_cands, context):
        # context_entities = self.get_unambiguous_entities(mentions_cands)
        contex_emb = self.encode_sentence(context)
        # context_entities_emb = self.get_context_entity_emb(context_entities)
        # ranks = {}
        all_mentions = list(mentions_cands.keys())
        max_prior_probs = {}
        feature_values = []
        for mention in mentions_cands:
            num_cands = len(mentions_cands[mention])
            for cand in mentions_cands[mention]:
                cand = cand.replace('_', ' ')
                # base features
                entity_prior = get_entity_prior(cand)
                prior_prob = get_prior_prob(cand, mention)
                if cand in max_prior_probs.keys():
                    max_prior_prob = max_prior_probs[cand]
                else:
                    max_prior_prob = max([get_prior_prob(cand, i) for i in all_mentions])

                # string similarity features
                mention_l = mention.lower()
                cand_edit_dist = get_edit_dist(mention_l, cand)
                mention_is_cand = cand == mention_l
                mention_in_cand = mention in cand
                is_start_or_end = cand.startswith(mention_l) or cand.endswith(mention_l)

                # contextual features
                candidate_enc = self.encode_entity(cand)
                context_sim = cos_sim(contex_emb, candidate_enc)
                # coherence = cos_sim(context_entities_emb, candidate_enc)

                feature_values.append([
                    mention, cand, entity_prior, prior_prob, max_prior_prob, num_cands,
                    cand_edit_dist, mention_is_cand, mention_in_cand, is_start_or_end,
                    context_sim
                ])
                
        return feature_values
