import numpy as np
import torch
import torch.nn.functional as f
from flair.data import Sentence
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def cos_sim(v1, v2):
    v1v2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    if v1v2 == 0:
        return 0
    else:
        return np.dot(v2, v1) / v1v2


class Base:
    def __init__(self, emb, cased=True, filter_stopwords=True):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = word_tokenize
        self.emb = emb
        self.cased = cased
        self.filter_stopwords = filter_stopwords

    def filter(self, tokens):
        return [w for w in tokens if not w.lower() in self.stop_words]

    def encode_sentence(self, s):
        if not self.cased:
            s = s.lower()
        words = self.tokenizer(s)
        if self.filter_stopwords:
            words = self.filter(words)
        enc = []
        for w in words:
            try:
                enc.append(self.emb.get_vector(w))
            except:
                pass
        if enc:
            return sum(enc) / len(enc)
        else:
            return np.zeros((self.emb.vector_size,))

    def rank(self, candidates, context):
        ranking = []
        for candidate in candidates:
            if candidates[candidate] is not None:
                score = cos_sim(context, candidates[candidate])
                ranking.append([candidate, score])
        return ranking

    def link(self, mention, context, candidates, top_only=True):
        # 2. Encode context-mention and candidates
        context = self.encode_sentence(context)
        candidate_enc = {i: self.encode_sentence(j) for i, j in candidates}

        # 3. Candidate Ranking
        ranking = self.rank(candidate_enc, context)
        ranking.sort(key=lambda x: x[1], reverse=True)
        if top_only:
            return ranking[0] if len(ranking) > 0 else ['NULL', 0]
        else:
            return ranking


class BaseWiki2Vec(Base):
    def __init__(self, emb, filter_stopwords=True, vector_size=100):
        super().__init__(emb, cased=False, filter_stopwords=filter_stopwords)
        self.vector_size = vector_size

    def encode_sentence(self, s, _):
        s = s.lower()
        words = self.tokenizer(s)
        if self.filter_stopwords:
            words = self.filter(words)
        enc = []
        for w in words:
            if self.emb.get_word(w.lower()) is not None:
                enc.append(self.emb.get_word_vector(w.lower()))
        if enc:
            return sum(enc) / len(enc)
        else:
            return np.zeros((self.vector_size,))

    def encode_entity(self, entity):
        entity = entity.replace('_', ' ')
        if self.emb.get_entity(entity) is not None:
            return self.emb.get_entity_vector(entity)
        else:
            return np.zeros(self.vector_size)

    def link(self, mention, context, candidates=None, top_only=True):
        # 2. Encode context-mention and candidates
        context = self.encode_sentence(context, mention)
        candidate_enc = {i: self.encode_entity(i) for i, _ in candidates}

        # 3. Candidate Ranking
        ranking = self.rank(candidate_enc, context)
        ranking.sort(key=lambda x: x[1], reverse=True)
        if top_only:
            return ranking[0] if len(ranking) > 0 else ['NULL', 0]
        else:
            return ranking


class BaseFlair(Base):
    def __init__(self, emb, cased=True, filter_stopwords=True):
        super().__init__(emb, cased, filter_stopwords)

    def encode_sentence(self, s):
        if not self.cased:
            s = s.lower()
        words = self.tokenizer(s)
        if self.filter_stopwords:
            words = self.filter(words)
        if words:
            sentence = Sentence(words)
            self.emb.embed(sentence)
            return sentence.get_embedding()
        else:
            return torch.zeros(self.emb.embedding_length)

    def rank(self, candidates, context):
        ranking = []
        for candidate in candidates.keys():
            score = f.cosine_similarity(context, candidates[candidate], dim=0)
            ranking.append([candidate, score.item()])
        return ranking
