# Evaluating Joint Word and Entity Embeddings for Entity Disambiguation
> Author: Atharva Tidke (htrf88)

## Setup

To retrain or test the implemented NED models,
- Download dependencies specified in `environment.yml` (Note: this may take up to 6.8GB of disk space).
- Run the `embeddings/download.py` script to download embeddings required for the experiments. Only Wikipedia2Vec embeddings are required to run the trained model, so like 15 onwards can be commented out to avoid downloading other embeddings. (Note: 100 dim Wikipedia2Vec embeddings require 3.6GB of disk space and same amount of memory on runtime. Downloading all embeddings will take up to 9GB of disk space)
- Ensure that `entity_prior.pkl`, `entity_anchors.pkl` are present in `data/GBRT`
- In addition to these the NLTK stopword corpus and BERT based NER model need to be downloaded for text preprocessing. The latter is downloaded automatically when required.

## NED Models
- The base and advanced NED models are included in the `src` folder in the `base.py` and `gbrt.py` files respectively. 
- The variant of the advanced model that uses SBERT embeddings is implemented in `transformers.py`.
- The variant of the advanced model that uses multi layer NN instead of GBRT for regression is included in `mlp.py`
- The original advanced implementation of the model can be accessed using a web interface by running `streamlit run main.py`

## Experiment Notebooks
- `0.CandidateGeneration.ipynb` includes the results of candidate generation experiments.
- `1.0` and `1.1` details the training procedure of the advanced NED models
- `2.Testing.ipynb` details the results of the NED experiments.
- `3.Evaluation.ipynb` details the feature importances of the trined models.