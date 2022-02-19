import numpy as np
from gensim.models import KeyedVectors
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wikipedia2vec import Wikipedia2Vec

from src.utils import cos_sim, get_entity_extract


class Base:
    def __init__(self, emb_path, cased=True, nouns_only=False):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = word_tokenize
        self.entity_desc_dict = None
        self.nouns_only = nouns_only
        try:
            self.cased = cased
            self.emb = KeyedVectors.load(emb_path)
            self.emb.get_word_vector = self.emb.get_vector
            self.vector_size = self.emb.vector_size
            self.is_wiki2vec = False
        except:
            self.cased = False
            self.emb = Wikipedia2Vec.load(emb_path)
            self.vector_size = self.emb.train_params['dim_size']
            self.is_wiki2vec = True


    def get_nouns(self, tokens):
        nouns = []
        for word, pos in pos_tag(tokens):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.extend(word.split(' '))
        return list(set(nouns))


    def filter(self, tokens):
        tokens = [w for w in tokens if not(w.lower() in self.stop_words)]
        tokens = [w for w in tokens if w.isalnum()]
        return self.get_nouns(tokens) if self.nouns_only else tokens


    def encode_entity(self, entity):
        if self.is_wiki2vec:
            entity = entity.replace('_', ' ')
            if self.emb.get_entity(entity) is not None:
                return self.emb.get_entity_vector(entity)
            else:
                return np.zeros(self.vector_size)
        else:
            if self.entity_desc_dict is None:
                desc = get_entity_extract(entity)
            else:
                desc = self.entity_desc_dict.get(entity, '')

            return self.encode_sentence(desc)


    def encode_sentence(self, s):
        words = self.filter(self.tokenizer(s if self.cased else s.lower()))
        emb, n = np.zeros(self.vector_size), 1
        for w in words:
            try:
                emb += self.emb.get_word_vector(w)
            except KeyError:
                if self.cased:
                    try:
                        emb += self.emb.get_word_vector(w.lower())
                    except:
                        pass

        return emb/n


    def rank(self, candidates, context):
        ranking = []
        for candidate in candidates:
            if candidates[candidate] is not None:
                score = cos_sim(context, candidates[candidate])
                ranking.append([candidate, score])
        return ranking


    def link(self, mention, context, candidates):
        context_enc = self.encode_sentence(context)
        pred, conf = 'NIL', 0
        for candidate in candidates:
            candidate_enc = self.encode_entity(candidate)
            score = cos_sim(candidate_enc, context_enc)
            if score > conf:
                pred = candidate
                conf = score
        return pred, conf
