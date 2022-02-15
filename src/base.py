import nltk
import string
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wikipedia2vec import Wikipedia2Vec

from src.utils import cos_sim, get_entity_extract


class Base:
    def __init__(self, emb_path, cased=False, is_wiki2vec=False, nouns_only=False):
        self.cased = cased
        self.is_wiki2vec = is_wiki2vec
        self.nouns_only = nouns_only
        self.stop_words = set(stopwords.words('english'))
        self.pun_table = str.maketrans('', '', string.punctuation)
        self.tokenizer = word_tokenize
        self.cached_entity_desc = None
        if is_wiki2vec:
            self.emb = Wikipedia2Vec.load(emb_path)
            self.vector_size = self.emb.train_params['dim_size']  
        else:
            self.emb = KeyedVectors.load(emb_path) if emb_path else None
            self.vector_size = self.emb.vector_size


    def get_nouns(self, tokens):
        nouns = []
        for word, pos in nltk.pos_tag(tokens):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.extend(word.split(' '))
        return list(set(nouns))
    
    def filter(self, tokens):
        words = [w.translate(self.pun_table) for w in tokens]
        words = [w for w in words if w.isalpha() and not(w.lower() in self.stop_words)]
        return words


    def encode_sentence(self, s):
        tokens = self.tokenizer(s if self.cased else s.lower())
        words = self.filter(tokens)
        if self.nouns_only: 
            words = [i for i in self.get_nouns(words)]
        
        # Encode
        enc, n = np.zeros(self.vector_size), 1
        if self.is_wiki2vec:
            for w in words:
                if self.emb.get_word(w.lower()) is not None:
                    enc += self.emb.get_word_vector(w.lower())
                    n += 1
        else:
            for w in words:
                try:
                    enc += self.emb.get_vector(w)
                except:
                    pass

        return enc/n


    def encode_entity(self, entity):
        if self.is_wiki2vec:
            entity = entity.replace('_', ' ')
            if self.emb.get_entity(entity) is not None:
                return self.emb.get_entity_vector(entity)
            else:
                return np.zeros(self.vector_size)
        else:
            if self.cached_entity_desc is None:
                desc = get_entity_extract(entity)
            else:
                desc = self.cached_entity_desc.get(entity, '')
            
            return self.encode_sentence(desc)                      


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
