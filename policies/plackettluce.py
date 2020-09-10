import numpy as np

def sample_rankings(log_scores, n_samples, cutoff=None, prob_per_rank=False):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings = np.empty((n_samples, ranking_len), dtype=np.int32)
  inv_rankings = np.empty((n_samples, n_docs), dtype=np.int32)
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)

  if cutoff:
    inv_rankings[:] = ranking_len

  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    cumprobs = np.cumsum(probs, axis=1)
    random_values = np.random.uniform(size=n_samples)
    greater_equal_mask = np.greater_equal(random_values[:,None], cumprobs)
    sampled_ind = np.sum(greater_equal_mask, axis=1)

    rankings[:, i] = sampled_ind
    inv_rankings[ind, sampled_ind] = i
    rankings_prob[:, i] = probs[ind, sampled_ind]
    log_scores[ind, sampled_ind] = np.NINF

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix
  else:
    return rankings, inv_rankings, rankings_prob

def sample_rankings_query(data_split,
                          qid,
                          log_scores,
                          n_samples,
                          cutoff=None,
                          prob_per_rank=False):
  s_i, e_i = data_split.query_range(qid)
  return sample_rankings(log_scores[s_i:e_i],
                         n_samples,
                         cutoff,
                         prob_per_rank)

def gradient_based_on_samples(sampled_rankings,
                              obs_prob,
                              log_scores,
                              rankings_prob,
                              cutoff=None):
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)
  result = np.zeros((n_docs, n_docs))
  log_scores = np.tile(log_scores[None,:], (n_samples, 1))

  cumulative_grad = np.zeros((n_samples, n_docs))
  cur_grad = np.zeros((n_docs, n_docs))

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  for i in range(ranking_len):
    cur_grad[:] = 0.
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    denom = np.log(np.sum(np.exp(log_scores), axis=1))
    cur_doc_prob = np.exp(log_scores[:,:] - denom[:, None])

    cur_grad[doc_ind, doc_ind] += np.mean(cur_doc_prob, axis=0)
    cur_grad -= np.mean(cur_doc_prob[:, :, None]*cur_doc_prob[:, None, :], axis=0)
    if i > 0:
      cur_grad += np.mean(cur_doc_prob[:, :, None]*cumulative_grad[:, None, :], axis=0)

    result += obs_prob[i]*cur_grad

    if i < n_docs - 1:
      cumulative_grad[sample_ind, sampled_rankings[:, i]] += 1
      cumulative_grad -= cur_doc_prob

      log_scores[sample_ind, sampled_rankings[:, i]] = np.NINF

  return result
