# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import itertools
from scipy.optimize import linprog

class OptimizedInterleaving(object):

  def __init__(self, data_split, model_rankings, inv_rankings, cutoff):
    self.data_split = data_split
    self.cutoff = cutoff
    self.scores =  1./(inv_rankings[0, :]+1.) - 1./(inv_rankings[1, :]+1.)
    (self.probs,
    self.prob_ranges,
    self.interleavings,
    self.ranges,
    self.n_interleavings,
    ) = optimize_policy(data_split, model_rankings, inv_rankings, cutoff)

  def sample(self, qid):
    s_i, e_i = self.data_split.query_range(qid)
    s_j, e_j = self.ranges[qid:qid+2]
    s_p, e_p = self.prob_ranges[qid:qid+2]
    n_inter = self.n_interleavings[qid]
    k = min(self.data_split.query_size(qid), self.cutoff)
    inters = np.reshape(self.interleavings[s_j:e_j], (n_inter, k))
    probs = self.probs[s_p:e_p]

    i = np.random.choice(n_inter, p=probs)

    return inters[i, :], self.scores[s_i:e_i]


def optimize_policy(data_split, model_rankings, inv_rankings, cutoff):
  n_queries = data_split.num_queries()
  # n_queries = 1000
  ranges = np.zeros(n_queries+1, dtype=np.int64)
  prob_ranges = np.zeros(n_queries+1, dtype=np.int64)
  n_interleavings = np.zeros(n_queries, dtype=np.int64)
  all_interleavings = []
  all_probs = []
  for qid in range(n_queries):
    s_i, e_i = data_split.query_range(qid)
    q_rankings = model_rankings[:,s_i:e_i]
    q_inv_rankings = inv_rankings[:,s_i:e_i]
    (q_probs,
     q_interleavings) = optimize_for_query(
                                  q_rankings, 
                                  q_inv_rankings,
                                  cutoff)
    all_interleavings.append(q_interleavings.flatten())
    all_probs.append(q_probs.flatten()/np.sum(q_probs))
    n_interleavings[qid] = q_interleavings.shape[0]
    ranges[qid+1] = ranges[qid] + all_interleavings[-1].shape[0]
    prob_ranges[qid+1] = prob_ranges[qid] + n_interleavings[qid]

  all_interleavings = np.concatenate(all_interleavings, axis=0).astype(np.int32)
  all_probs = np.concatenate(all_probs, axis=0)

  return all_probs, prob_ranges, all_interleavings, ranges, n_interleavings

def update_index(i, ranking, interleaving):
  while ranking[i] in interleaving:
    i += 1
  return i

def optimize_for_query(rankings, inv_rankings, cutoff):
  n_docs = rankings.shape[1]
  k = np.minimum(cutoff, n_docs)
  if n_docs == 1:
    probs = np.ones((1,1), dtype=np.float64)
    interleavings = np.zeros((1,1), dtype=np.float64)
    return probs, interleavings
  else:
    interleavings = {(): (0,0)}
    for i in range(k):
      next_interleavings = {}
      for cur_inter, (i0, i1) in interleavings.items():
        i0 = update_index(i0, rankings[0, :], cur_inter)
        i1 = update_index(i1, rankings[1, :], cur_inter)
        if rankings[0, i0] == rankings[1, i1]:
          cur_inter = cur_inter + (rankings[0, i0],)
          next_interleavings[cur_inter] = (i0+1, i1+1)
        else:
          new_inter = cur_inter + (rankings[0, i0],)
          next_interleavings[new_inter] = (i0+1, i1)

          new_inter = cur_inter + (rankings[1, i1],)
          next_interleavings[new_inter] = (i0, i1+1)

      interleavings = next_interleavings

  interleavings = np.stack(list(interleavings.keys()), axis=0)
  scores = inv_rankings[0,:] - inv_rankings[1,:]
  interleaving_scores = scores[interleavings]
  n_interleaving = interleavings.shape[0]

  A = np.concatenate((interleaving_scores,
                      np.ones((n_interleaving, 1), dtype=np.float64)),
                     axis=1)
  c = np.zeros(n_interleaving, dtype=np.float64)
  b = np.zeros(k+1, dtype=np.float64)
  b[-1] = 1.
  bounds = [(0.,1.)]

  res = linprog(c, A_eq=A.T, b_eq=b, bounds=bounds, options={'disp': False})

  probs = res.x

  return probs, interleavings
