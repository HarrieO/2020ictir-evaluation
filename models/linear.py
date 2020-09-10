# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np

class LinearModel(object):

  def __init__(self, num_features):
    self.weights = np.zeros(num_features, dtype=np.float64)

  def score(self, feature_matrix):
    return np.dot(feature_matrix, self.weights)

  def score_query(self, qid, data_split, additional_feat=None):
    query_feat = data_split.query_feat(qid)
    if additional_feat is not None:
      s_i, e_i = data_split.query_range(qid)
      query_feat = np.concatenate(
                        (query_feat,
                        additional_feat[s_i:e_i, :]),
                        axis=1,
                      )
    return self.score(query_feat)

  def score_data_split(self, data_split, additional_feat=None):
    data_feat = data_split.feature_matrix
    if additional_feat is not None:
      data_feat = np.concatenate(
                        (data_feat,
                        additional_feat),
                        axis=1,
                      )
    return self.score(data_feat)

  def gradient_update(self, doc_weights, doc_feat,
                      additional_feat=None,
                      learning_rate=0.01):
    doc_weights = np.nan_to_num(doc_weights)
    if additional_feat is not None:
      gradient = np.zeros_like(self.weights)
      n_non_add = doc_feat.shape[1]
      gradient[:n_non_add] = np.sum(doc_weights[:, None]*doc_feat, axis=0)
      gradient[n_non_add:] = np.sum(doc_weights[:, None]*additional_feat, axis=0)
    else:
      gradient = np.sum(doc_weights[:, None]*doc_feat, axis=0)
    self.weights += np.nan_to_num(gradient*learning_rate)