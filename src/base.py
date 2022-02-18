import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.utils import cos_sim, get_entity_extract
from wikipedia2vec import Wikipedia2Vec


class Base:
    def __init__(self, emb_path='', cased=True):
        self.cased = cased
        self.emb = KeyedVectors.load(emb_path) if emb_path else None
        self.entity_desc_dict = None
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = word_tokenize

    def filter(self, tokens):
        return [w for w in tokens if not w.lower() in self.stop_words]
    
    def encode_entity(self, entity):
        if self.entity_desc_dict is None:
            desc = get_entity_extract(entity)
        else:
            desc = self.entity_desc_dict.get(entity, '')
            
        return self.encode_sentence(desc)

    def encode_sentence(self, s):
        words = self.tokenizer(s) if self.cased else self.tokenizer(s.lower())
        words = self.filter(words)
        emb, n = np.zeros(self.emb.vector_size), 1
        for w in words:
            try:
                emb += self.emb.get_vector(w)
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


class BaseWiki2Vec(Base):
    def __init__(self, emb_path):
        super().__init__()
        self.emb = Wikipedia2Vec.load(emb_path)
        self.vector_size = self.emb.train_params['dim_size']

    def encode_sentence(self, s):
        words = self.filter(self.tokenizer(s.lower()))
        emb, n = np.zeros(self.vector_size), 1
        for w in words:
            if self.emb.get_word(w.lower()) is not None:
                emb += self.emb.get_word_vector(w.lower())
                n += 1
        return emb/n

    def encode_entity(self, entity):
        entity = entity.replace('_', ' ')
        if self.emb.get_entity(entity) is not None:
            return self.emb.get_entity_vector(entity)
        else:
            return np.zeros(self.vector_size)
