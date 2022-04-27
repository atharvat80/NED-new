import os
import numpy as np
import torch
import torch.nn as nn

from src.base import Base
from src.gbrt import get_edit_dist, get_entity_prior, get_max_prior_prob, get_prior_prob
from src.utils import cosine_similarity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chkpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'NN_ranker.pt')

"""
Entity Ranking using deep neural network
"""

class MLP(nn.Module):
    def __init__(self, num_in_features=11, dropout=0.3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_in_features, 1000),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(1000, 1),
        )
    
    def forward(self, x):
        return self.layers(x.float())
        

class MLPRanker(Base):
    def __init__(self, emb_path, two_step=False, cased=False, nouns_only=True):
        super().__init__(emb_path, cased=cased, nouns_only=nouns_only)
        self.two_step = two_step
        self.model = MLP().to(device)
        self.model.load_state_dict(torch.load(chkpt_path))
        self.model.eval()

    def encode_context_entities(self, context_entities):
        emb, n = np.zeros(self.vector_size), 1
        for i in context_entities:
            emb += self.encode_entity(i)
            n += 1
        return emb/n

    def link(self, mentions_cands, context):
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

                X.append([
                    # base features
                    candidate, 
                    get_prior_prob(candidate, mention), 
                    get_entity_prior(candidate),
                    max_prob[candidate],
                    num_cands,
                    # Add string similarity
                    get_edit_dist(ment, cand),
                    int(ment == cand),
                    int(ment in cand),
                    int(cand.startswith(cand) or cand.endswith(ment)),
                    # Add context similarity
                    cosine_similarity(cand_emb, context_emb),
                    # Add coherence score
                    cosine_similarity(cand_emb, context_ent_emb)
                ])
                    
            # Add rank
            X.sort(key=lambda x: x[-1] + x[-2], reverse=True)
            X = [j + [i + 1] for i, j in enumerate(X)]

            # Predict
            pred, conf = 'NIL', 0
            for i in X:
                with torch.no_grad():
                    c = self.model(torch.Tensor(i[1:]).to(device)).item()
                if c > conf:
                    pred = i[0]
                    conf = c
            predictions.append([mention, pred, conf])

            # Update context entity embedding (two-step)
            if self.two_step:
                context_ent_emb += self.encode_entity(pred)
                context_ent_emb /= 2

        return predictions