import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from src.gbrt import get_edit_dist, get_entity_prior, get_max_prior_prob, get_prior_prob, load_model
from src.utils import get_entity_extract, cosine_similarity


class BaseTRF:
    def __init__(self):
        self.vector_size = 384
        self.entity_desc_dict = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)

    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_entity(self, entity):
        if self.entity_desc_dict is None:
            desc = get_entity_extract(entity)
        else:
            desc = self.entity_desc_dict.get(entity, '')
        return self.encode_sentence(desc)

    def encode_sentence(self, s):
        if s:
            # Tokenize sentences
            encoded_input = self.tokenizer([s], padding=True, truncation=True, return_tensors='pt').to(self.device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            # Perform pooling. In this case, max pooling.
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            return sentence_embeddings[0].detach().cpu().numpy()
        else:
            return np.zeros((self.vector_size,))

    def link(self, mention, context, candidates):
        context_enc = self.encode_sentence(context)
        pred, conf = 'NIL', 0
        for candidate in candidates:
            candidate_enc = self.encode_entity(candidate)
            score = cosine_similarity(candidate_enc, context_enc)
            if score > conf:
                pred = candidate
                conf = score
        return pred, conf


# reference - https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


class GBRT_TRF(BaseTRF):
    def __init__(self, ranker_path=None, two_step=False):
        super().__init__()
        self.two_step = two_step
        if ranker_path is not None:
            self.ranker = load_model(ranker_path)
        
    def encode_context_entities(self, context_entities):
        emb, n = np.zeros(self.vector_size), 1
        for i in context_entities:
            emb += self.encode_entity(i)
            n += 1
        return emb/n

    def link(self, mentions_cands, context):
        n_features = self.ranker.n_features_in_

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
                        int(ment == cand),
                        int(ment in cand),
                        int(cand.startswith(cand) or cand.endswith(ment))
                    ]
                # Add context similarity
                if n_features >= 9:
                    feat_values.append(cosine_similarity(cand_emb, context_emb))
                # Add coherence score
                if n_features >= 10:
                    feat_values.append(cosine_similarity(cand_emb, context_ent_emb))
                X.append(feat_values)

            # Add rank
            if n_features == 11:
                X.sort(key=lambda x: x[-1] + x[-2], reverse=True)
                X = [j + [i + 1] for i, j in enumerate(X)]

            # Predict
            pred, conf = 'NIL', 0
            for i in X:
                c = self.ranker.predict(np.array([i[1:]]))[0]
                if c > conf:
                    pred = i[0]
                    conf = c
            predictions.append([mention, pred, conf])

            # Update context entity embedding (two-step)
            if self.two_step and n_features >= 10:
                context_ent_emb += self.encode_entity(pred)
                context_ent_emb /= 2

        return predictions
