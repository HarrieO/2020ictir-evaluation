# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np

def rank_and_invert(scores, tiebreakers=None):
  n_docs = scores.shape[0]
  noise = np.random.uniform(size=n_docs)
  if tiebreakers is not None:
    rank_ind = np.lexsort((noise, tiebreakers, scores))[::-1]
  else:
    rank_ind = np.lexsort((noise, scores))[::-1]
  inverted = np.empty(n_docs, dtype=rank_ind.dtype)
  inverted[rank_ind] = np.arange(n_docs)
  return rank_ind, inverted

def data_split_rank_and_invert(scores, data_split, tiebreakers=None):
  ranking = np.zeros(data_split.num_docs(), dtype=np.int64)
  inverted = np.zeros(data_split.num_docs(), dtype=np.int64)
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = scores[s_i:e_i]
    if tiebreakers is not None:
      q_tiebreakers = tiebreakers[s_i:e_i]
      (ranking[s_i:e_i],
       inverted[s_i:e_i]) = rank_and_invert(
                              q_scores,
                              tiebreakers=q_tiebreakers
                            )
    else:
      (ranking[s_i:e_i],
       inverted[s_i:e_i]) = rank_and_invert(q_scores)
  return ranking, inverted

def data_split_model_rank_and_invert(model, data_split):
  scores = model_score(model, data_split.feature_matrix)
  return data_split_rank_and_invert(scores, data_split)

def data_split_rank_and_invert_tiebreak_model(scores, model, data_split):
  model_scores = model_score(model, data_split.feature_matrix)
  return data_split_rank_and_invert(scores, data_split, tiebreakers=model_scores)

def model_score(model, doc_feat):
  return np.dot(model, doc_feat.T)

def model_rank_and_invert(model, doc_feat):
  scores = model_score(model, doc_feat)
  return rank_and_invert(scores)

def many_models_rank_and_invert(models, doc_feat):
  n_models = models.shape[0]
  n_docs = doc_feat.shape[0]
  model_scores = model_score(models, doc_feat)
  rankings = np.empty((n_models, n_docs), dtype=np.int64)
  inv_rankings = np.empty((n_models, n_docs), dtype=np.int64)
  for i in range(n_models):
    (rankings[i, :],
     inv_rankings[i,:])  = rank_and_invert(model_scores[i, :])
  return rankings, inv_rankings

def many_models_data_split_rank_and_invert(models, data_split):
  n_models = models.shape[0]
  n_docs = data_split.num_docs()
  rankings = np.empty((n_models, n_docs), dtype=np.int64)
  inv_rankings = np.empty((n_models, n_docs), dtype=np.int64)
  for i in range(n_models):
    (rankings[i, :],
     inv_rankings[i,:])  = data_split_model_rank_and_invert(models[i, :], data_split)
  return rankings, inv_rankings